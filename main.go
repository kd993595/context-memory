package main

import (
	"bytes"
	"context"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"unicode/utf8"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/table"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	"github.com/charmbracelet/lipgloss"
	duckdb "github.com/marcboeker/go-duckdb"
	"github.com/spf13/cobra"
)

const (
	appName       = "mdmemory" // directory name under $XDG_DATA_HOME
	dbFileName    = "entries.duckdb"
	markdownStyle = "dark"

	// embeddingDim is the size of the vector stored in entries.embedding.
	embeddingDim = 384

	// searchLimit is the maximum number of rows a search returns.
	searchLimit = 10

	// queryPrefix is prepended to search text before embedding. BGE recommends
	// an instruction prefix only for short-query -> long-passage retrieval.
	// Here both sides are short subject lines, so it stays empty.
	queryPrefix = ""

	// searchBarHeight is the height of the search line above the panes.
	searchBarHeight = 1
)

var paneStyle = lipgloss.NewStyle().
	BorderStyle(lipgloss.RoundedBorder()).
	BorderForeground(lipgloss.Color("62"))

var (
	statusStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("241")) // dim note next to the search box
	errStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("203")) // search errors
)

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

// dataDir returns $XDG_DATA_HOME/<appName>, falling back to ~/.local/share per
// the XDG spec, and creates it if needed.
func dataDir() (string, error) {
	base := os.Getenv("XDG_DATA_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolving home dir: %w", err)
		}
		base = filepath.Join(home, ".local", "share")
	}
	dir := filepath.Join(base, appName)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("creating data dir: %w", err)
	}
	return dir, nil
}

// dbPath returns the DuckDB file location inside dataDir.
func dbPath() (string, error) {
	dir, err := dataDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, dbFileName), nil
}

// openDB opens (creating if necessary) the DuckDB file, loads the vss
// extension, and ensures the schema exists. No HNSW index is stored in the
// file: persistent HNSW indexes are experimental and crashed on reopen, so
// the index is rebuilt in memory on every startup (see buildSearchIndex).
func openDB() (*sql.DB, error) {
	path, err := dbPath()
	if err != nil {
		return nil, err
	}

	// The init function runs on EVERY new connection in the pool. That matters
	// because LOAD and SET are per-connection, and database/sql may open
	// several connections.
	connector, err := duckdb.NewConnector(path, func(execer driver.ExecerContext) error {
		for _, q := range []string{
			"INSTALL vss", // no-op once installed (needs network the first time)
			"LOAD vss",
		} {
			if _, err := execer.ExecContext(context.Background(), q, nil); err != nil {
				return fmt.Errorf("%s: %w", q, err)
			}
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("opening duckdb at %s: %w", path, err)
	}

	db := sql.OpenDB(connector)

	schema := []string{
		`CREATE SEQUENCE IF NOT EXISTS entries_id_seq START 1`,
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS entries (
			id        BIGINT PRIMARY KEY DEFAULT nextval('entries_id_seq'),
			date      DATE NOT NULL DEFAULT current_date,
			subject   VARCHAR(255),
			content   TEXT,
			embedding FLOAT[%d]
		)`, embeddingDim),
	}
	for _, q := range schema {
		if _, err := db.Exec(q); err != nil {
			db.Close()
			return nil, fmt.Errorf("initializing schema: %w", err)
		}
	}
	return db, nil
}

// searchTable is the in-memory table holding only (id, embedding). It lives in
// a second, in-memory database attached to the same DuckDB instance, so its
// HNSW index never touches the on-disk file. Everything else (date, subject,
// content) stays on disk and is joined in by id when a search runs.
const searchTable = "mem.vecs"

// buildSearchIndex copies (id, embedding) from the on-disk entries table into
// an in-memory table and builds the HNSW index on it. Call it once at startup,
// before the UI starts. Entries added later with the new command show up the next time the
// browser is started.
//
// If the table is built but the index cannot be, searches still work (DuckDB
// falls back to a brute-force scan). That is returned as warn rather than
// printed, because this runs while the UI owns the screen; err is for failures
// that make searching impossible.
func buildSearchIndex(db *sql.DB) (warn error, err error) {
	stmts := []string{
		`ATTACH IF NOT EXISTS ':memory:' AS mem`,
		`CREATE OR REPLACE TABLE ` + searchTable + ` AS
			SELECT id, embedding FROM entries WHERE embedding IS NOT NULL`,
	}
	for _, q := range stmts {
		if _, e := db.Exec(q); e != nil {
			return nil, fmt.Errorf("building in-memory search table: %w", e)
		}
	}

	if _, e := db.Exec(`CREATE INDEX vecs_embedding_idx ON ` + searchTable +
		` USING HNSW (embedding) WITH (metric = 'cosine')`); e != nil {
		return fmt.Errorf("HNSW index not built, searches will scan every row: %w", e), nil
	}
	return nil, nil
}

// scanRows converts (id, date, subject) result rows into table rows.
func scanRows(res *sql.Rows) ([]table.Row, error) {
	var rows []table.Row
	for res.Next() {
		var (
			id      int64
			date    sql.NullTime
			subject sql.NullString
		)
		if err := res.Scan(&id, &date, &subject); err != nil {
			return nil, err
		}
		d := ""
		if date.Valid {
			d = date.Time.Format("2006-01-02")
		}
		rows = append(rows, table.Row{strconv.FormatInt(id, 10), d, subject.String})
	}
	return rows, res.Err()
}

// columnPadding is the horizontal space the table adds around each column's
// text (one cell on each side), so a column of Width w occupies w+2 cells.
const columnPadding = 2

// tableColumns returns the table layout for a pane paneWidth cells wide. The
// ID, Date (and, for search results, Score) columns have fixed widths; Subject
// gets everything that is left, so the row always fills the pane. Without
// scored the columns are ID, Date, Subject; with it, ID, Date, Score, Subject.
// Score sits before Subject so a very narrow pane clips the subject text
// rather than the score.
func tableColumns(scored bool, paneWidth int) []table.Column {
	cols := []table.Column{
		{Title: "ID", Width: 4},
		{Title: "Date", Width: 11},
	}
	if scored {
		cols = append(cols, table.Column{Title: "Score", Width: 6})
	}
	used := 0
	for _, c := range cols {
		used += c.Width + columnPadding
	}
	subject := max(10, paneWidth-used-columnPadding)
	return append(cols, table.Column{Title: "Subject", Width: subject})
}

// loadRows fetches the full list shown in the left table (no content/vector).
func loadRows(db *sql.DB) ([]table.Row, error) {
	res, err := db.Query(`SELECT id, date, subject FROM entries ORDER BY date DESC, id DESC`)
	if err != nil {
		return nil, err
	}
	defer res.Close()
	return scanRows(res)
}

// searchHit is one search result: an entry plus its cosine similarity to the
// query (1 = identical direction).
type searchHit struct {
	ID         int64
	Date       string // YYYY-MM-DD
	Subject    string
	Similarity float64
}

// searchHits returns the limit entries whose embeddings are closest (by cosine
// distance) to vec, best match first, each with its cosine similarity. It is
// the single implementation of the ranking: the browser shows its results in
// the table (see searchRows) and the search subcommand prints them as JSON.
//
// The query has two parts. The CTE is the exact shape the HNSW optimizer
// looks for (ORDER BY array_cosine_distance(<indexed column>, <constant
// vector>) LIMIT n over a single table), so it runs as an index scan over the
// in-memory searchTable and yields the n nearest ids. Those ids are then
// joined to the on-disk entries table for date and subject. A join does not
// preserve the CTE's ordering, so the outer query re-computes the similarity
// for just those n rows to sort the final result. That same value is selected
// as the score column, so showing the score costs no extra calculation: it
// replaces the ORDER BY computation rather than adding to it. Similarity is
// 1 - cosine distance, so sorting it descending gives the same order as the
// distance ascending. The vector is inlined as a literal; it is generated from
// float32 values only, never from user text.
func searchHits(db *sql.DB, vec []float32, limit int) ([]searchHit, error) {
	lit := fmt.Sprintf("'%s'::FLOAT[%d]", vectorLiteral(vec), embeddingDim)
	q := fmt.Sprintf(`
		WITH hits AS (
			SELECT id FROM %[1]s
			ORDER BY array_cosine_distance(embedding, %[2]s)
			LIMIT %[3]d
		)
		SELECT e.id, e.date, e.subject,
			array_cosine_similarity(v.embedding, %[2]s) AS score
		FROM hits h
		JOIN entries e ON e.id = h.id
		JOIN %[1]s v ON v.id = h.id
		ORDER BY score DESC`,
		searchTable, lit, limit)
	res, err := db.Query(q)
	if err != nil {
		return nil, err
	}
	defer res.Close()

	var hits []searchHit
	for res.Next() {
		var (
			h       searchHit
			date    sql.NullTime
			subject sql.NullString
			score   sql.NullFloat64
		)
		if err := res.Scan(&h.ID, &date, &subject, &score); err != nil {
			return nil, err
		}
		if date.Valid {
			h.Date = date.Time.Format("2006-01-02")
		}
		h.Subject = subject.String
		h.Similarity = score.Float64
		hits = append(hits, h)
	}
	return hits, res.Err()
}

// searchRows runs searchHits and converts the result into table rows laid out
// as tableColumns(true, ...): {id, date, score, subject}.
func searchRows(db *sql.DB, vec []float32, limit int) ([]table.Row, error) {
	hits, err := searchHits(db, vec, limit)
	if err != nil {
		return nil, err
	}
	rows := make([]table.Row, 0, len(hits))
	for _, h := range hits {
		rows = append(rows, table.Row{
			strconv.FormatInt(h.ID, 10), h.Date, fmt.Sprintf("%.3f", h.Similarity), h.Subject,
		})
	}
	return rows, nil
}

// searchResultMsg carries the rows for the table after a search (or after
// clearing a search, in which case query is empty and rows is the full list).
type searchResultMsg struct {
	id    int // search id; stale results are discarded
	query string
	rows  []table.Row
	err   error

	// scored is true when rows carry the similarity column (see tableColumns).
	scored bool
}

// searchCmd embeds the query and runs the vector search off the event loop.
func searchCmd(id int, db *sql.DB, emb *embedder, query string) tea.Cmd {
	return func() tea.Msg {
		vec, err := emb.Embed(context.Background(), queryPrefix+query)
		if err != nil {
			return searchResultMsg{id: id, query: query, err: err}
		}
		rows, err := searchRows(db, vec, searchLimit)
		return searchResultMsg{id: id, query: query, rows: rows, err: err, scored: true}
	}
}

// listAllCmd restores the unfiltered list.
func listAllCmd(id int, db *sql.DB) tea.Cmd {
	return func() tea.Msg {
		rows, err := loadRows(db)
		return searchResultMsg{id: id, rows: rows, err: err}
	}
}

func oneLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// ---------------------------------------------------------------------------
// Key bindings
// ---------------------------------------------------------------------------

type keyMap struct {
	Up         key.Binding
	Down       key.Binding
	ScrollUp   key.Binding
	ScrollDown key.Binding
	Wheel      key.Binding
	Search     key.Binding
	Submit     key.Binding // display-only: handled while the search box has focus
	Cancel     key.Binding // display-only: handled while the search box has focus
	Help       key.Binding
	Quit       key.Binding
}

func newKeyMap() keyMap {
	return keyMap{
		Up:         key.NewBinding(key.WithKeys("up", "k"), key.WithHelp("↑/k", "previous entry")),
		Down:       key.NewBinding(key.WithKeys("down", "j"), key.WithHelp("↓/j", "next entry")),
		ScrollUp:   key.NewBinding(key.WithKeys("pgup"), key.WithHelp("pgup", "scroll preview up")),
		ScrollDown: key.NewBinding(key.WithKeys("pgdown"), key.WithHelp("pgdn", "scroll preview down")),
		// Display-only: the mouse wheel is handled in Update, not via key matching.
		Wheel:  key.NewBinding(key.WithKeys("wheel"), key.WithHelp("mouse wheel", "scroll preview (over right pane)")),
		Search: key.NewBinding(key.WithKeys("/", "tab"), key.WithHelp("/", "search")),
		Submit: key.NewBinding(key.WithKeys("enter"), key.WithHelp("enter", "run search (empty = show all)")),
		Cancel: key.NewBinding(key.WithKeys("esc"), key.WithHelp("esc", "leave search box")),
		Help:   key.NewBinding(key.WithKeys("?"), key.WithHelp("?", "toggle help")),
		Quit:   key.NewBinding(key.WithKeys("q", "ctrl+c"), key.WithHelp("q/ctrl+c", "quit")),
	}
}

func (k keyMap) ShortHelp() []key.Binding {
	return []key.Binding{k.Up, k.Down, k.Search, k.ScrollUp, k.ScrollDown, k.Help, k.Quit}
}

func (k keyMap) FullHelp() [][]key.Binding {
	return [][]key.Binding{
		{k.Up, k.Down},
		{k.ScrollUp, k.ScrollDown, k.Wheel},
		{k.Search, k.Submit, k.Cancel},
		{k.Help, k.Quit},
	}
}

// ---------------------------------------------------------------------------
// Markdown renderer (shared, built once per wrap width)
// ---------------------------------------------------------------------------

type mdRenderer struct {
	mu sync.Mutex
	r  *glamour.TermRenderer
}

func newMDRenderer(wrap int) *mdRenderer {
	r, err := glamour.NewTermRenderer(
		glamour.WithStandardStyle(markdownStyle),
		glamour.WithWordWrap(wrap),
	)
	if err != nil {
		r = nil // Render will fall back to plain text
	}
	return &mdRenderer{r: r}
}

func (m *mdRenderer) Render(in string) string {
	if m == nil || m.r == nil {
		return in
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	out, err := m.r.Render(in)
	if err != nil {
		return in
	}
	return out
}

// ---------------------------------------------------------------------------
// Async content loading
// ---------------------------------------------------------------------------

type entryRenderedMsg struct {
	id         int // request id; stale results are discarded
	content    string
	keepOffset bool
	offset     int
}

// renderEntryCmd fetches an entry's content from DuckDB and renders it as
// markdown, off the Bubble Tea event loop.
func renderEntryCmd(reqID int, db *sql.DB, entryID int64, r *mdRenderer, keepOffset bool, offset int) tea.Cmd {
	return func() tea.Msg {
		var content sql.NullString
		err := db.QueryRow(`SELECT content FROM entries WHERE id = ?`, entryID).Scan(&content)

		var raw string
		switch {
		case err != nil:
			raw = fmt.Sprintf("# Error\nCould not load entry `%d`.\n\n`%v`", entryID, err)
		case !content.Valid || content.String == "":
			raw = "*(This entry has no content)*"
		default:
			raw = content.String
		}
		return entryRenderedMsg{
			id:         reqID,
			content:    r.Render(raw),
			keepOffset: keepOffset,
			offset:     offset,
		}
	}
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

type model struct {
	db       *sql.DB
	embedder *embedder
	search   textinput.Model
	table    table.Model
	viewport viewport.Model
	help     help.Model
	keys     keyMap
	ready    bool

	renderer      *mdRenderer
	rendererWidth int

	scored    bool // table rows carry the Score column (they are search results)
	leftWidth int  // inner width of the left pane, used to size the Subject column

	loadedRow int // row whose entry was last requested (-1 = none yet)
	reqID     int // incremented per request; only the latest result is applied

	query     string // active search text ("" = full list)
	searchID  int    // incremented per search; only the latest result is applied
	status    string // shown to the right of the search box
	statusErr bool

	width, height int
	leftOuter     int
	paneHeight    int
}

const emptyMessage = "\n  No entries yet.\n\n  Add one with:  " + appName + " new -s \"subject\" -t \"content\""

func initialModel(db *sql.DB, rows []table.Row, emb *embedder) model {
	t := table.New(
		table.WithColumns(tableColumns(false, 0)), // widened to the pane by resize
		table.WithRows(rows),
		table.WithFocused(true),
	)

	t.KeyMap = table.KeyMap{
		LineUp:   key.NewBinding(key.WithKeys("up", "k"), key.WithHelp("↑/k", "up")),
		LineDown: key.NewBinding(key.WithKeys("down", "j"), key.WithHelp("↓/j", "down")),
	}

	s := table.DefaultStyles()
	s.Selected = s.Selected.
		Foreground(lipgloss.Color("229")).
		Background(lipgloss.Color("62")).
		Bold(false)
	t.SetStyles(s)

	ti := textinput.New()
	ti.Prompt = "/ "
	ti.Placeholder = "search entries by meaning (press / to type)"
	ti.CharLimit = 255

	return model{
		db:        db,
		embedder:  emb,
		search:    ti,
		table:     t,
		help:      help.New(),
		keys:      newKeyMap(),
		loadedRow: -1,
	}
}

func (m model) Init() tea.Cmd { return nil }

func (m *model) resize() tea.Cmd {
	if m.width == 0 || m.height == 0 {
		return nil
	}

	m.help.Width = m.width
	helpHeight := lipgloss.Height(m.help.View(m.keys))

	m.search.Width = max(10, m.width/2-lipgloss.Width(m.search.Prompt)-1)

	m.leftOuter = m.width / 3
	rightOuter := m.width - m.leftOuter

	m.paneHeight = m.height - helpHeight - searchBarHeight - 2
	if m.paneHeight < 3 {
		m.paneHeight = 3
	}

	leftWidth := m.leftOuter - 2
	rightWidth := rightOuter - 2

	m.leftWidth = leftWidth
	m.table.SetColumns(tableColumns(m.scored, m.leftWidth)) // Subject fills the pane
	m.table.SetWidth(leftWidth)
	m.table.SetHeight(m.paneHeight)

	if !m.ready {
		m.viewport = viewport.New(rightWidth, m.paneHeight)
		m.viewport.KeyMap = viewport.KeyMap{
			PageUp:   key.NewBinding(key.WithKeys("pgup")),
			PageDown: key.NewBinding(key.WithKeys("pgdown")),
		}
		if len(m.table.Rows()) == 0 {
			m.viewport.SetContent(emptyMessage)
		}
		m.ready = true
	} else {
		m.viewport.Width = rightWidth
		m.viewport.Height = m.paneHeight
	}

	wrap := rightWidth - 2
	if wrap < 10 {
		wrap = 10
	}
	if m.renderer == nil || wrap != m.rendererWidth {
		m.renderer = newMDRenderer(wrap)
		m.rendererWidth = wrap

		if m.loadedRow != -1 {
			return m.loadRow(true, false)
		}
	}
	return nil
}

// loadRow starts an asynchronous fetch+render of the selected entry.
func (m *model) loadRow(keepOffset, showLoading bool) tea.Cmd {
	if len(m.table.Rows()) == 0 {
		return nil
	}
	selected := m.table.SelectedRow()
	if len(selected) == 0 { // the id is always the first cell
		return nil
	}
	entryID, err := strconv.ParseInt(selected[0], 10, 64)
	if err != nil {
		return nil
	}

	m.loadedRow = m.table.Cursor()
	m.reqID++

	if showLoading {
		m.viewport.SetContent(fmt.Sprintf("\n  Loading entry %d…", entryID))
		m.viewport.GotoTop()
	}

	return renderEntryCmd(m.reqID, m.db, entryID, m.renderer, keepOffset, m.viewport.YOffset)
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.KeyMsg:
		// While the search box has focus, keys are text input, so q, ? and
		// friends must not trigger their list-mode actions.
		if m.search.Focused() {
			switch msg.String() {
			case "ctrl+c":
				return m, tea.Quit
			case "esc":
				m.search.Blur()
				return m, nil
			case "enter":
				m.search.Blur()
				return m, m.startSearch(strings.TrimSpace(m.search.Value()))
			case "up", "down": // navigate results without leaving the box
				var cmd tea.Cmd
				m.table, cmd = m.table.Update(msg)
				cmds = append(cmds, cmd)
			case "pgup", "pgdown":
				if m.ready {
					var cmd tea.Cmd
					m.viewport, cmd = m.viewport.Update(msg)
					cmds = append(cmds, cmd)
				}
			default:
				var cmd tea.Cmd
				m.search, cmd = m.search.Update(msg)
				return m, cmd
			}
			cmds = append(cmds, m.syncSelection())
			return m, tea.Batch(cmds...)
		}

		switch {
		case key.Matches(msg, m.keys.Quit): // q and ctrl+c
			return m, tea.Quit
		case key.Matches(msg, m.keys.Help):
			m.help.ShowAll = !m.help.ShowAll
			return m, m.resize()
		case key.Matches(msg, m.keys.Search):
			cmds = append(cmds, m.search.Focus())
			return m, tea.Batch(cmds...)
		}

		// Remaining keys: table consumes up/down, viewport consumes pgup/pgdn.
		var cmd tea.Cmd
		m.table, cmd = m.table.Update(msg)
		cmds = append(cmds, cmd)
		if m.ready {
			m.viewport, cmd = m.viewport.Update(msg)
			cmds = append(cmds, cmd)
		}

	case tea.MouseMsg:
		if m.ready && msg.X >= m.leftOuter {
			var cmd tea.Cmd
			m.viewport, cmd = m.viewport.Update(msg)
			cmds = append(cmds, cmd)
		}

	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		cmds = append(cmds, m.resize())

	case searchResultMsg:
		if msg.id != m.searchID {
			return m, nil
		}
		if msg.err != nil {
			m.status = "Search failed: " + oneLine(msg.err.Error())
			m.statusErr = true
			m.loadedRow = -1 // re-render the current row's preview
			break
		}
		m.statusErr = false
		m.setTableRows(msg.rows, msg.scored)
		m.table.SetCursor(0)
		m.loadedRow = -1 // syncSelection below loads the new first row
		switch {
		case msg.query == "":
			m.status = ""
		default:
			m.status = fmt.Sprintf("%d result(s) for %q", len(msg.rows), msg.query)
		}
		if len(msg.rows) == 0 {
			if msg.query == "" {
				m.viewport.SetContent(emptyMessage)
			} else {
				m.viewport.SetContent("\n  No matching entries.")
			}
			m.viewport.GotoTop()
		}

	case entryRenderedMsg:
		if msg.id != m.reqID {
			return m, nil
		}
		m.viewport.SetContent(msg.content)
		if msg.keepOffset {
			m.viewport.SetYOffset(msg.offset)
		} else {
			m.viewport.GotoTop()
		}
		return m, nil
	}

	cmds = append(cmds, m.syncSelection())
	return m, tea.Batch(cmds...)
}

// setTableRows replaces the table contents and switches between the plain and
// scored column layouts. The table indexes columns by cell position when it
// renders, so a row must never have more cells than there are columns:
// the wider layout is installed before the rows, the narrower one after.
func (m *model) setTableRows(rows []table.Row, scored bool) {
	m.scored = scored
	if scored {
		m.table.SetColumns(tableColumns(true, m.leftWidth))
		m.table.SetRows(rows)
		return
	}
	m.table.SetRows(rows)
	m.table.SetColumns(tableColumns(false, m.leftWidth))
}

// syncSelection loads the selected entry's preview if the cursor moved. The
// left pane already shows the new cursor on this frame; the content loads in
// the background while the right pane shows a loading message.
func (m *model) syncSelection() tea.Cmd {
	if m.ready && len(m.table.Rows()) > 0 && m.table.Cursor() != m.loadedRow {
		return m.loadRow(false, true)
	}
	return nil
}

// startSearch runs a search for q, or restores the full list when q is empty.
func (m *model) startSearch(q string) tea.Cmd {
	m.searchID++
	m.reqID++ // discard any in-flight preview render
	m.query = q
	m.statusErr = false

	if q == "" {
		m.status = "Loading…"
		return listAllCmd(m.searchID, m.db)
	}
	m.status = "Searching…"
	return searchCmd(m.searchID, m.db, m.embedder, q)
}

func (m model) View() string {
	if !m.ready {
		return "\n  Initializing..."
	}

	leftPane := paneStyle.
		Width(m.leftOuter - 2).
		Height(m.paneHeight).
		Render(m.table.View())

	rightPane := paneStyle.
		Width(m.viewport.Width).
		Height(m.paneHeight).
		Render(m.viewport.View())

	panes := lipgloss.JoinHorizontal(lipgloss.Top, leftPane, rightPane)
	return lipgloss.JoinVertical(lipgloss.Left, m.searchBar(), panes, m.help.View(m.keys))
}

// searchBar renders the single search line: the input plus a dim status note.
func (m model) searchBar() string {
	bar := m.search.View()
	if m.status != "" {
		st := statusStyle
		if m.statusErr {
			st = errStyle
		}
		if room := m.width - lipgloss.Width(bar) - 2; room > 0 {
			bar += "  " + st.MaxWidth(room).Render(oneLine(m.status))
		}
	}
	return lipgloss.NewStyle().MaxWidth(m.width).Render(bar)
}

// vectorLiteral formats a vector as a DuckDB array literal, e.g. [0.1,0.2].
func vectorLiteral(v []float32) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(f), 'g', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}

// insertEntry adds a row dated today, including its embedding, and returns
// its new id. The vector is passed as text and cast to FLOAT[embeddingDim] in
// SQL, which avoids depending on driver support for array parameters.
func insertEntry(db *sql.DB, subject, content string, embedding []float32) (int64, error) {
	var id int64
	err := db.QueryRow(
		fmt.Sprintf(`INSERT INTO entries (date, subject, content, embedding)
			VALUES (current_date, ?, ?, ?::FLOAT[%d]) RETURNING id`, embeddingDim),
		subject, content, vectorLiteral(embedding),
	).Scan(&id)
	return id, err
}

// updateContent replaces the content column of the entry with the given id.
// The embedding is computed from the subject only, so it stays valid and the
// embedding model is not needed.
func updateContent(db *sql.DB, id int64, content string) error {
	if !utf8.ValidString(content) {
		return fmt.Errorf("replacement content is not valid UTF-8")
	}
	res, err := db.Exec(`UPDATE entries SET content = ? WHERE id = ?`, content, id)
	if err != nil {
		return err
	}
	if n, rerr := res.RowsAffected(); rerr == nil && n == 0 {
		return fmt.Errorf("no entry with id %d", id)
	}
	return nil
}

// loadContent returns the content column of the entry with the given id. A
// NULL content comes back as the empty string.
func loadContent(db *sql.DB, id int64) (string, error) {
	var content sql.NullString
	err := db.QueryRow(`SELECT content FROM entries WHERE id = ?`, id).Scan(&content)
	if errors.Is(err, sql.ErrNoRows) {
		return "", fmt.Errorf("no entry with id %d", id)
	}
	if err != nil {
		return "", err
	}
	return content.String, nil
}

// parseRowID parses a positional <rowid> argument.
func parseRowID(s string) (int64, error) {
	id, err := strconv.ParseInt(s, 10, 64)
	if err != nil || id < 1 {
		return 0, fmt.Errorf("invalid rowid %q: must be a positive integer", s)
	}
	return id, nil
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

// newRootCmd builds the root command. With no subcommand it starts the
// interactive browser; the other operations are subcommands that each define
// their own local flags.
func newRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   appName,
		Short: "Browse markdown entries stored in DuckDB",
		Long: `Browse markdown entries stored in DuckDB.

With no subcommand, starts the interactive browser. Use the new, update, load
and search subcommands to add an entry, replace an entry's content, print an
entry's content, or find the entry closest to a text, without starting the UI.`,
		Example: `  ` + appName + `
  ` + appName + ` new -s "Meeting notes" -t "# Agenda\n- item one"
  ` + appName + ` new -s "Meeting notes" -f notes.md
  ` + appName + ` update 12 "# New content"
  ` + appName + ` update 12 -f notes.md
  ` + appName + ` load 12
  ` + appName + ` load 12 -f notes.md
  ` + appName + ` search "meeting notes"
  ` + appName + ` search "meeting notes" -C
  ` + appName + ` search "meeting notes" -f result.json`,
		Args:          cobra.NoArgs, // also makes a mistyped subcommand an error
		SilenceErrors: true,         // main prints the error itself
		RunE: func(cmd *cobra.Command, args []string) error {
			cmd.SilenceUsage = true // runtime errors shouldn't print usage
			return runUI()
		},
	}
	root.AddCommand(newEntryCmd(), updateEntryCmd(), loadEntryCmd(), searchEntryCmd())
	return root
}

// newEntryCmd is "new": insert an entry dated today.
func newEntryCmd() *cobra.Command {
	var (
		subject string
		content string
		file    string
	)

	cmd := &cobra.Command{
		Use:   "new -s <subject> (-t <content> | -f <file>)",
		Short: "Insert a new entry dated today",
		Long: `Insert a single entry dated today and exit without starting the UI.
The content is given with -t/--content or read from a file with -f/--file.
The subject is embedded locally with bge-small-en-v1.5 (downloaded on first use).`,
		Example: `  ` + appName + ` new -s "Meeting notes" -t "# Agenda\n- item one"
  ` + appName + ` new -s "Meeting notes" -f notes.md`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			contentSet := cmd.Flags().Changed("content")
			fileSet := cmd.Flags().Changed("file")
			switch {
			case contentSet && fileSet:
				return fmt.Errorf("use either -t/--content or -f/--file, not both")
			case !contentSet && !fileSet:
				return fmt.Errorf("one of -t/--content or -f/--file is required")
			}
			if strings.TrimSpace(subject) == "" {
				return fmt.Errorf("subject must not be empty")
			}
			if len([]rune(subject)) > 255 {
				return fmt.Errorf("subject is too long (%d characters, max 255)", len([]rune(subject)))
			}

			cmd.SilenceUsage = true // flags are valid; runtime errors shouldn't print usage

			// Read the file before opening the database or loading the model,
			// so a bad path fails fast.
			if fileSet {
				b, err := os.ReadFile(file)
				if err != nil {
					return fmt.Errorf("reading %s: %w", file, err)
				}
				content = string(b)
			}
			if !utf8.ValidString(content) {
				return fmt.Errorf("content is not valid UTF-8")
			}

			ctx, stop := signal.NotifyContext(context.Background(),
				os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)
			defer stop()

			// Setup (open the database, load the embedding model) must finish
			// before anything is embedded or inserted, so a model failure never
			// leaves a row without a vector. Progress goes to stderr.
			a, err := setup(ctx, false, func(p progress) {
				fmt.Fprintf(os.Stderr, "[%d/%d] %s\n", p.Step, p.Total, p.Text)
			})
			if err != nil {
				return err
			}
			defer func() {
				if cerr := a.Close(); cerr != nil {
					fmt.Fprintf(os.Stderr, "warning: %v\n", cerr)
				}
			}()

			vec, err := a.emb.Embed(ctx, subject)
			if err != nil {
				return err
			}

			id, err := insertEntry(a.db, subject, content, vec)
			if err != nil {
				return fmt.Errorf("inserting entry: %w", err)
			}
			fmt.Fprintf(cmd.OutOrStdout(), "Created entry %d\n", id)
			return nil
		},
	}

	f := cmd.Flags()
	f.StringVarP(&subject, "subject", "s", "", "subject of the new entry (max 255 characters)")
	f.StringVarP(&content, "content", "t", "", "markdown content of the new entry (or use -f)")
	f.StringVarP(&file, "file", "f", "", "read the markdown content of the new entry from this `file` (or use -t)")
	_ = cmd.MarkFlagRequired("subject")

	return cmd
}

// updateEntryCmd is "update": replace the content column of an entry.
func updateEntryCmd() *cobra.Command {
	var file string

	cmd := &cobra.Command{
		Use:   "update <rowid> [text]",
		Short: "Replace the content of an entry",
		Long: `Replace the content of the entry with the given rowid and exit without
starting the UI. The new content is the text argument or, with -f/--file, the
contents of a file. The subject and its embedding are left unchanged.`,
		Example: `  ` + appName + ` update 12 "# New content"
  ` + appName + ` update 12 -f notes.md`,
		Args: cobra.RangeArgs(1, 2),
		RunE: func(cmd *cobra.Command, args []string) error {
			id, err := parseRowID(args[0])
			if err != nil {
				return err
			}

			var text string
			if cmd.Flags().Changed("file") {
				if len(args) != 1 {
					return fmt.Errorf("give the replacement content either as text or with -f/--file, not both")
				}
				b, err := os.ReadFile(file)
				if err != nil {
					cmd.SilenceUsage = true
					return fmt.Errorf("reading %s: %w", file, err)
				}
				text = string(b)
			} else {
				if len(args) != 2 {
					return fmt.Errorf("replacement text is required (or use -f/--file)")
				}
				text = args[1]
			}

			cmd.SilenceUsage = true // arguments are valid; runtime errors shouldn't print usage

			// Only the database is needed: the embedding comes from the
			// subject, which is unchanged, so the model is not loaded.
			db, err := openDB()
			if err != nil {
				return err
			}
			uerr := updateContent(db, id, text)
			if cerr := db.Close(); cerr != nil {
				if uerr == nil {
					return fmt.Errorf("closing database: %w", cerr)
				}
				fmt.Fprintf(os.Stderr, "warning: closing database: %v\n", cerr)
			}
			if uerr != nil {
				return fmt.Errorf("updating entry %d: %w", id, uerr)
			}
			fmt.Fprintf(cmd.OutOrStdout(), "Updated entry %d\n", id)
			return nil
		},
	}

	cmd.Flags().StringVarP(&file, "file", "f", "", "read the replacement content from this `file`")

	return cmd
}

// loadEntryCmd is "load": print an entry's content or write it to a file.
func loadEntryCmd() *cobra.Command {
	var file string

	cmd := &cobra.Command{
		Use:   "load <rowid>",
		Short: "Print the content of an entry",
		Long: `Print the content of the entry with the given rowid and exit without starting
the UI. With -f/--file the content is written to that file instead (an existing
file is overwritten). Printing adds a trailing newline if the content lacks one;
the file receives the content exactly as stored.`,
		Example: `  ` + appName + ` load 12
  ` + appName + ` load 12 -f notes.md`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			id, err := parseRowID(args[0])
			if err != nil {
				return err
			}
			fileSet := cmd.Flags().Changed("file")
			if fileSet && file == "" {
				return fmt.Errorf("-f/--file needs a filename")
			}

			cmd.SilenceUsage = true // arguments are valid; runtime errors shouldn't print usage

			db, err := openDB()
			if err != nil {
				return err
			}
			content, lerr := loadContent(db, id)
			if cerr := db.Close(); cerr != nil {
				if lerr == nil {
					return fmt.Errorf("closing database: %w", cerr)
				}
				fmt.Fprintf(os.Stderr, "warning: closing database: %v\n", cerr)
			}
			if lerr != nil {
				return fmt.Errorf("loading entry %d: %w", id, lerr)
			}

			out := cmd.OutOrStdout()
			if fileSet {
				if err := os.WriteFile(file, []byte(content), 0o644); err != nil {
					return fmt.Errorf("writing %s: %w", file, err)
				}
				fmt.Fprintf(out, "Wrote entry %d to %s\n", id, file)
				return nil
			}
			if content == "" {
				return nil
			}
			fmt.Fprint(out, content)
			if !strings.HasSuffix(content, "\n") {
				fmt.Fprintln(out)
			}
			return nil
		},
	}

	cmd.Flags().StringVarP(&file, "file", "f", "", "write the content to this `file` instead of printing it")

	return cmd
}

// searchResult is the JSON printed by the search subcommand. Content is a
// pointer so that it is left out entirely unless requested, while an entry
// whose content is empty still prints "content": "".
type searchResult struct {
	ID         int64   `json:"id"`
	Date       string  `json:"date"`
	Subject    string  `json:"subject"`
	Similarity float64 `json:"similarity"`
	Content    *string `json:"content,omitempty"`
}

// marshalSearchResult encodes r as indented JSON with a trailing newline.
// HTML escaping is off so that <, > and & in subjects and content stay readable.
func marshalSearchResult(r searchResult) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(r); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// searchEntryCmd is "search": find the single entry whose subject is closest
// in meaning to a text and print it as JSON.
func searchEntryCmd() *cobra.Command {
	var (
		withContent bool
		file        string
	)

	cmd := &cobra.Command{
		Use:   "search <text> [-C] [-f <file>]",
		Short: "Print the entry whose subject best matches a text, as JSON",
		Long: `Find the one entry whose subject is closest in meaning to the given text and
print it as JSON with the keys id, date, subject and similarity (the cosine
similarity, 1 = identical direction), then exit without starting the UI. It
ranks entries exactly as the browser's search does.

With -C/--content the JSON also has a "content" key holding the entry's content
column. With -f/--file the JSON, including content, is written to that file
(an existing file is overwritten) and only a confirmation is printed.

Progress messages go to stderr, so stdout is only the JSON. The command fails
if there are no entries with an embedding to match against.`,
		Example: `  ` + appName + ` search "meeting notes"
  ` + appName + ` search "meeting notes" -C
  ` + appName + ` search "meeting notes" -f result.json`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			text := strings.TrimSpace(strings.Join(args, " "))
			if text == "" {
				return fmt.Errorf("search text must not be empty")
			}
			fileSet := cmd.Flags().Changed("file")
			if fileSet && file == "" {
				return fmt.Errorf("-f/--file needs a filename")
			}

			cmd.SilenceUsage = true // arguments are valid; runtime errors shouldn't print usage

			ctx, stop := signal.NotifyContext(context.Background(),
				os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)
			defer stop()

			// Same startup as the browser, including the in-memory HNSW index,
			// so the ranking is the one the browser uses. Progress goes to
			// stderr to keep stdout clean for the JSON.
			a, err := setup(ctx, true, func(p progress) {
				fmt.Fprintf(os.Stderr, "[%d/%d] %s\n", p.Step, p.Total, p.Text)
			})
			if err != nil {
				return err
			}
			defer func() {
				if cerr := a.Close(); cerr != nil {
					fmt.Fprintf(os.Stderr, "warning: %v\n", cerr)
				}
			}()
			if a.warning != nil {
				fmt.Fprintf(os.Stderr, "warning: %s\n", oneLine(a.warning.Error()))
			}

			vec, err := a.emb.Embed(ctx, queryPrefix+text)
			if err != nil {
				return err
			}
			hits, err := searchHits(a.db, vec, 1)
			if err != nil {
				return fmt.Errorf("searching: %w", err)
			}
			if len(hits) == 0 {
				return fmt.Errorf("no entries to search")
			}
			hit := hits[0]

			res := searchResult{
				ID:         hit.ID,
				Date:       hit.Date,
				Subject:    hit.Subject,
				Similarity: hit.Similarity,
			}
			// The file always includes the content; stdout only with -C.
			if withContent || fileSet {
				content, err := loadContent(a.db, hit.ID)
				if err != nil {
					return fmt.Errorf("loading entry %d: %w", hit.ID, err)
				}
				res.Content = &content
			}
			data, err := marshalSearchResult(res)
			if err != nil {
				return fmt.Errorf("encoding JSON: %w", err)
			}

			out := cmd.OutOrStdout()
			if fileSet {
				if err := os.WriteFile(file, data, 0o644); err != nil {
					return fmt.Errorf("writing %s: %w", file, err)
				}
				fmt.Fprintf(out, "Wrote entry %d to %s\n", hit.ID, file)
				return nil
			}
			_, err = out.Write(data)
			return err
		},
	}

	f := cmd.Flags()
	f.BoolVarP(&withContent, "content", "C", false, "include the entry's content in the JSON as \"content\"")
	f.StringVarP(&file, "file", "f", "", "write the JSON, including content, to this `file` instead of printing it")

	return cmd
}

func main() {
	if err := newRootCmd().Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

// app holds everything the program needs that is expensive to create. It is
// built once by setup and lives until the process exits.
type app struct {
	db      *sql.DB
	emb     *embedder
	warning error // non-fatal setup problem to show once the UI is up
}

// Close releases the model and closes the database.
func (a *app) Close() error {
	if a == nil {
		return nil
	}
	a.emb.Close()
	if a.db == nil {
		return nil
	}
	if err := a.db.Close(); err != nil {
		return fmt.Errorf("closing database: %w", err)
	}
	return nil
}

// progress describes where startup is: step Step of Total, doing Text.
type progress struct {
	Step, Total int
	Text        string
}

// setup is the startup routine shared by the browser and the new command: it opens DuckDB,
// loads the embedding model (downloading it on first run), and, for the
// browser, builds the in-memory search index. report may be nil; it is called
// from the calling goroutine and must not write to the terminal while the UI
// is running.
func setup(ctx context.Context, withIndex bool, report func(progress)) (*app, error) {
	total := 2
	if withIndex {
		total = 3
	}
	step := 0
	say := func(text string) {
		step++
		if report != nil {
			report(progress{Step: step, Total: total, Text: text})
		}
	}

	say("Opening database…")
	db, err := openDB()
	if err != nil {
		return nil, err
	}
	a := &app{db: db}
	fail := func(err error) (*app, error) {
		_ = a.Close() // release whatever was opened; the original error wins
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}

	say("Loading embedding model…")
	logf := func(msg string) { // e.g. the first-run download notice
		if report != nil {
			report(progress{Step: step, Total: total, Text: msg})
		}
	}
	emb, err := newEmbedder(ctx, logf)
	if err != nil {
		return fail(err)
	}
	a.emb = emb

	if withIndex {
		say("Building search index…")
		warn, err := buildSearchIndex(db)
		if err != nil {
			return fail(err)
		}
		a.warning = warn
	}
	return a, nil
}

// uiSetup runs setup for the browser and also loads the entry list.
func uiSetup(ctx context.Context, report func(progress)) (*app, []table.Row, error) {
	a, err := setup(ctx, true, report)
	if err != nil {
		return nil, nil, err
	}
	rows, err := loadRows(a.db)
	if err != nil {
		_ = a.Close()
		return nil, nil, fmt.Errorf("loading entries: %w", err)
	}
	return a, rows, nil
}

// progressMsg and setupDoneMsg carry startup results into the loading screen.
type progressMsg progress

type setupDoneMsg struct {
	app  *app
	rows []table.Row
	err  error
}

// loadingModel is the first screen the browser shows. It displays startup
// progress and, once setup finishes, hands over to the main model.
type loadingModel struct {
	spinner       spinner.Model
	prog          progress
	width, height int
}

func newLoadingModel() loadingModel {
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	return loadingModel{spinner: sp, prog: progress{Step: 0, Total: 3, Text: "Starting…"}}
}

func (m loadingModel) Init() tea.Cmd { return m.spinner.Tick }

func (m loadingModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		if msg.String() == "ctrl+c" || msg.String() == "q" {
			return m, tea.Quit
		}
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd
	case progressMsg:
		m.prog = progress(msg)
	case setupDoneMsg:
		if msg.err != nil {
			return m, tea.Quit // runUI reports the error after the screen is restored
		}
		next := initialModel(msg.app.db, msg.rows, msg.app.emb)
		if msg.app.warning != nil {
			next.status = oneLine(msg.app.warning.Error())
			next.statusErr = true
		}
		if m.width > 0 {
			// The window size already arrived; replay it so the main model can lay out.
			updated, cmd := next.Update(tea.WindowSizeMsg{Width: m.width, Height: m.height})
			return updated, cmd
		}
		return next, nil
	}
	return m, nil
}

func (m loadingModel) View() string {
	const barWidth = 30
	filled := 0
	if m.prog.Total > 0 {
		// The current step is in progress, so only earlier steps count as done.
		filled = barWidth * max(0, m.prog.Step-1) / m.prog.Total
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)

	body := fmt.Sprintf("%s %s\n\n%s  %d/%d",
		m.spinner.View(), oneLine(m.prog.Text),
		statusStyle.Render(bar), max(1, m.prog.Step), m.prog.Total)
	if m.width > 0 && m.height > 0 {
		return lipgloss.Place(m.width, m.height, lipgloss.Center, lipgloss.Center, body)
	}
	return "\n  " + body
}

// runUI starts the interactive Bubble Tea browser. It shows a loading screen
// while setup runs in the background, then switches to the main view.
// SIGINT, SIGTERM and SIGHUP (e.g. the terminal window being closed) stop the
// program through the normal return path, so the app is always closed cleanly.
func runUI() error {
	ctx, stop := signal.NotifyContext(context.Background(),
		os.Interrupt, syscall.SIGTERM, syscall.SIGHUP)
	defer stop()

	p := tea.NewProgram(
		newLoadingModel(),
		tea.WithContext(ctx),
		tea.WithAltScreen(),
		tea.WithMouseCellMotion(),
	)

	// Setup runs in our own goroutine (not a tea.Cmd) so that after the UI
	// exits we can wait for it and release whatever it created, even if the
	// user quit mid-load.
	setupCtx, cancelSetup := context.WithCancel(ctx)
	defer cancelSetup()

	type result struct {
		app *app
		err error
	}
	finished := make(chan result, 1)
	go func() {
		a, rows, err := uiSetup(setupCtx, func(pr progress) { p.Send(progressMsg(pr)) })
		finished <- result{app: a, err: err}
		p.Send(setupDoneMsg{app: a, rows: rows, err: err})
	}()

	_, runErr := p.Run()

	// The UI is gone: cancel any download still running, wait for setup to
	// finish, and shut everything down.
	cancelSetup()
	res := <-finished
	if cerr := res.app.Close(); cerr != nil {
		fmt.Fprintf(os.Stderr, "warning: %v\n", cerr)
	}

	if runErr != nil && !errors.Is(runErr, tea.ErrProgramKilled) {
		return runErr
	}
	if res.err != nil && !errors.Is(res.err, context.Canceled) {
		return res.err
	}
	return nil
}
