import re



class EvidenceStore:
    """保存单次研究中的证据，按 URL 去重并分配编号。"""

    def __init__(self):
        # 证据编号 → 证据内容
        self._evidence_by_id: dict[str, dict[str, str]] = {}

        # 网页 URL → 证据编号
        self._id_by_url: dict[str, str] = {}

    @classmethod
    def from_records(cls, records: list[dict]) -> "EvidenceStore":
        """恢复已有编号；拒绝重复 URL、缺号或被改坏的数据。"""
        if not isinstance(records, list):
            raise ValueError("检查点证据必须是列表")
        store = cls()
        for index, item in enumerate(records, start=1):
            if (not isinstance(item, dict)
                    or set(item) != {"id", "title", "url", "content"}
                    or not all(isinstance(value, str) for value in item.values())
                    or item["id"] != f"E{index}"
                    or not item["url"] or item["url"] != item["url"].strip()
                    or item["url"] in store._id_by_url):
                raise ValueError("检查点证据编号、URL 或内容不正确")
            store._evidence_by_id[item["id"]] = item.copy()
            store._id_by_url[item["url"]] = item["id"]
        return store

    def add(self, item: dict[str, str]) -> dict[str, str]:
        """添加一条搜索结果，返回带编号的证据。"""

        url = item["url"].strip()
        if not url:
            raise ValueError("证据 URL 不能为空")

        # 同一个 URL 已经出现过，直接返回原有证据
        if url in self._id_by_url:
            evidence_id = self._id_by_url[url]
            return self._evidence_by_id[evidence_id].copy()

        # 新 URL：分配编号并保存
        evidence_id = f"E{len(self._evidence_by_id) + 1}"

        evidence = {
            "id": evidence_id,
            "title": item["title"],
            "url": url,
            "content": item["content"],
        }

        self._evidence_by_id[evidence_id] = evidence
        self._id_by_url[url] = evidence_id

        return evidence.copy()

    def get(self, evidence_id: str) -> dict[str, str] | None:
        """根据编号查询证据，不存在时返回 None。"""

        evidence = self._evidence_by_id.get(evidence_id)

        if evidence is None:
            return None

        return evidence.copy()

    def all(self) -> list[dict[str, str]]:
        """返回当前保存的全部证据。"""

        return [
            evidence.copy()
            for evidence in self._evidence_by_id.values()
        ]

    def find_unknown_citations(self, text: str) -> list[str]:
        """找出正文中引用了、但证据库里不存在的编号。"""

        cited_ids = re.findall(r"\[(E[1-9][0-9]*)\]", text)

        unknown_ids = []

        for evidence_id in cited_ids:
            if self.get(evidence_id) is None:
                if evidence_id not in unknown_ids:
                    unknown_ids.append(evidence_id)

        return unknown_ids
