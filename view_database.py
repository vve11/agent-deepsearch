"""本地 SQLite 只读查看器：python view_database.py，无第三方依赖。"""
import json
import sqlite3
import tkinter as tk
from contextlib import closing
from pathlib import Path
from tkinter import messagebox, ttk


DB_PATH = Path(__file__).resolve().parent / "data" / "research.db"
PAGE_SIZE = 100


def connect_readonly(path: Path) -> sqlite3.Connection:
    # mode=ro 由 SQLite 强制禁止写入；文件不存在时也不会创建空数据库。
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    connection.execute("PRAGMA query_only = ON")
    return connection


def quote_identifier(name: str) -> str:
    # 参数占位符不能用于表名；这里的名称来自数据库元数据，并正确转义。
    return '"' + name.replace('"', '""') + '"'


def format_value(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return repr(value)
    text = str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return json.dumps(parsed, ensure_ascii=False, indent=2)
    except (ValueError, TypeError):
        pass
    return text


class DatabaseViewer:
    def __init__(self, root: tk.Tk, path: Path):
        self.root, self.path = root, path
        self.page = 0
        self.rows = []
        self.columns = []
        root.title("研究助手 · SQLite 数据库查看器（只读）")
        root.geometry("1180x760")
        root.minsize(780, 500)
        ttk.Label(root, text=f"只读连接：{path}", padding=10).pack(anchor="w")
        bar = ttk.Frame(root, padding=(10, 0, 10, 8))
        bar.pack(fill="x")
        ttk.Label(bar, text="数据表：").pack(side="left")
        self.table = ttk.Combobox(bar, state="readonly", width=28)
        self.table.pack(side="left", padx=5)
        self.table.bind("<<ComboboxSelected>>", lambda event: self.change_table())
        ttk.Button(bar, text="刷新", command=self.refresh).pack(side="left", padx=5)
        self.previous = ttk.Button(bar, text="上一页", command=lambda: self.turn_page(-1))
        self.previous.pack(side="left", padx=5)
        self.next = ttk.Button(bar, text="下一页", command=lambda: self.turn_page(1))
        self.next.pack(side="left", padx=5)
        self.status = ttk.Label(bar)
        self.status.pack(side="left", padx=10)

        panes = ttk.Panedwindow(root, orient="vertical")
        panes.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        grid_frame = ttk.Frame(panes)
        panes.add(grid_frame, weight=3)
        self.grid = ttk.Treeview(grid_frame, show="headings", selectmode="browse")
        horizontal = ttk.Scrollbar(grid_frame, orient="horizontal", command=self.grid.xview)
        vertical = ttk.Scrollbar(grid_frame, orient="vertical", command=self.grid.yview)
        self.grid.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.grid.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        grid_frame.rowconfigure(0, weight=1)
        grid_frame.columnconfigure(0, weight=1)
        self.grid.bind("<<TreeviewSelect>>", self.show_record)

        details_frame = ttk.Frame(panes)
        panes.add(details_frame, weight=2)
        ttk.Label(details_frame, text="表结构 / 完整记录（选择上方一行查看；支持选中文字复制）").pack(anchor="w")
        scroll = ttk.Scrollbar(details_frame)
        scroll.pack(side="right", fill="y")
        self.details = tk.Text(details_frame, wrap="word", state="disabled", yscrollcommand=scroll.set)
        self.details.pack(fill="both", expand=True)
        scroll.configure(command=self.details.yview)
        self.refresh()

    def set_details(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def refresh(self):
        try:
            with closing(connect_readonly(self.path)) as connection:
                tables = [row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_schema WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )]
            self.table["values"] = tables
            if self.table.get() not in tables:
                self.table.set(tables[0] if tables else "")
                self.page = 0
            self.load_page()
        except sqlite3.Error as error:
            self.status.configure(text="读取失败")
            messagebox.showerror("无法读取数据库", str(error), parent=self.root)

    def change_table(self):
        self.page = 0
        self.refresh()

    def turn_page(self, delta):
        self.page = max(0, self.page + delta)
        self.refresh()

    def load_page(self):
        self.grid.delete(*self.grid.get_children())
        self.rows = []
        table = self.table.get()
        if not table:
            self.grid["columns"] = ()
            self.status.configure(text="暂无数据表")
            self.previous.configure(state="disabled")
            self.next.configure(state="disabled")
            self.set_details("数据库中暂无数据表。请通过正常程序初始化数据库；查看器不会创建表。")
            return
        name = quote_identifier(table)
        with closing(connect_readonly(self.path)) as connection:
            # 短只读事务使本页计数和内容一致；读完立即关闭，不长期占用连接。
            connection.execute("BEGIN")
            schema = connection.execute(f"PRAGMA table_info({name})").fetchall()
            count = connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            self.page = min(self.page, max(0, (count - 1) // PAGE_SIZE))
            primary = sorted((col for col in schema if col[5]), key=lambda col: col[5])
            order = ", ".join(quote_identifier(col[1]) for col in primary) or "rowid"
            self.rows = connection.execute(
                f"SELECT * FROM {name} ORDER BY {order} LIMIT ? OFFSET ?",
                (PAGE_SIZE, self.page * PAGE_SIZE),
            ).fetchall()
        self.columns = [col[1] for col in schema]
        ids = [f"column_{i}" for i in range(len(self.columns))]
        self.grid["columns"] = ids
        for column_id, label in zip(ids, self.columns):
            self.grid.heading(column_id, text=label)
            self.grid.column(column_id, width=190, minwidth=90, stretch=False)
        for index, row in enumerate(self.rows):
            cells = ["NULL" if value is None else str(value).replace("\n", " ") for value in row]
            self.grid.insert("", "end", iid=str(index), values=[v[:160] + ("…" if len(v) > 160 else "") for v in cells])
        self.status.configure(text=f"共 {count} 条 · 第 {self.page + 1}/{max(1, (count + PAGE_SIZE - 1) // PAGE_SIZE)} 页")
        self.previous.configure(state="normal" if self.page else "disabled")
        self.next.configure(state="normal" if (self.page + 1) * PAGE_SIZE < count else "disabled")
        self.set_details("表结构：" + table + "\n\n" + "\n".join(
            f"{col[1]}    {col[2]}" + ("    PRIMARY KEY" if col[5] else "")
            + ("    NOT NULL" if col[3] else "") for col in schema
        ))

    def show_record(self, event=None):
        selected = self.grid.selection()
        if not selected:
            return
        index = int(selected[0])
        if index < len(self.rows):
            self.set_details("\n\n".join(
                f"【{name}】\n{format_value(value)}" for name, value in zip(self.columns, self.rows[index])
            ))


def main():
    root = tk.Tk()
    DatabaseViewer(root, DB_PATH)
    root.mainloop()


if __name__ == "__main__":
    main()
