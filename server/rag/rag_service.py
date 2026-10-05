import asyncio
import hashlib
import re

from config import settings
from services.security import validate_identifier

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class RAGService:
    """RAG 服务 - 基于 ChromaDB 的文档检索"""

    def __init__(self):
        self.chroma_path = settings.CHROMADB_PATH
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path) if chromadb else None
        self._embed_model = None

    @property
    def embed_model(self):
        if self._embed_model is None and SentenceTransformer:
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embed_model

    async def ingest_document(
        self,
        project_id: str,
        file_path: str,
        file_type: str = "txt",
        doc_type: str = "novel",
    ) -> dict:
        """导入文档到 RAG"""

        try:
            safe_project_id = validate_identifier(project_id, "项目 ID")
        except ValueError:
            # RAG collection names must be safe even when called by an older
            # integration that did not validate the route parameter.
            safe_project_id = hashlib.sha1(str(project_id).encode("utf-8")).hexdigest()[:24]

        # Parsing docx/txt and embedding are synchronous libraries. Keep them
        # off the event loop so a large import cannot stall WebSocket progress.
        parser = self._parse_docx if file_type.lower() == "docx" else self._parse_txt
        raw_text = await asyncio.to_thread(parser, file_path)
        if len(raw_text) > settings.MAX_SCRIPT_TEXT_CHARS:
            raise ValueError("RAG 文档超过长度限制")

        # 智能分块
        chunks = self._smart_chunk(raw_text, doc_type)

        stored = False
        if self.chroma_client is not None and chunks:
            await asyncio.to_thread(self._store_chunks, safe_project_id, doc_type, chunks)
            stored = True

        # 提取角色和场景
        all_characters = set()
        all_scenes = set()
        for chunk in chunks:
            all_characters.update(chunk.get("characters", []))
            if chunk.get("location"):
                all_scenes.add(chunk["location"])

        return {
            "total_chunks": len(chunks),
            "characters_extracted": list(all_characters),
            "scenes_extracted": list(all_scenes),
            "stored": stored,
        }

    def _store_chunks(self, project_id: str, doc_type: str, chunks: list[dict]) -> None:
        collection = self.chroma_client.get_or_create_collection(
            name=f"project_{project_id}",
            metadata={"doc_type": doc_type},
        )
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for index, chunk in enumerate(chunks):
            text = str(chunk.get("text") or "")
            digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
            ids.append(f"chunk_{index:04d}_{digest}")
            documents.append(text)
            metadatas.append(
                {
                    "chunk_type": str(chunk.get("type", "narrative")),
                    "chapter": str(chunk.get("chapter", "")),
                    "characters": ",".join(str(item) for item in chunk.get("characters", [])),
                    "location": str(chunk.get("location", "")),
                    "position": index,
                }
            )
        # upsert makes repeated imports idempotent instead of failing on fixed
        # chunk IDs. Batch writes also reduce Chroma overhead.
        if hasattr(collection, "upsert"):
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        else:  # pragma: no cover - compatibility with very old Chroma clients
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

    async def query_context(
        self,
        project_id: str,
        query: str,
        n_results: int = 3,
        filter_type: str = None,
    ) -> list[str]:
        """查询相关上下文"""

        if self.chroma_client is None:
            return []
        try:
            collection = self.chroma_client.get_collection(f"project_{project_id}")
        except Exception:
            return []

        where_filter = {"chunk_type": filter_type} if filter_type else None

        results = collection.query(
            query_texts=[query],
            n_results=max(1, min(int(n_results or 3), 20)),
            where=where_filter,
        )

        if results and results["documents"]:
            return results["documents"][0]
        return []

    async def query_character_context(self, project_id: str, character_name: str) -> list[str]:
        """查询角色相关上下文"""

        if self.chroma_client is None:
            return []
        try:
            collection = self.chroma_client.get_collection(f"project_{project_id}")
        except Exception:
            return []

        results = collection.query(
            query_texts=[character_name],
            n_results=5,
            where={"characters": {"$contains": character_name}},
        )

        if results and results["documents"]:
            return results["documents"][0]
        return []

    def _parse_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_docx(self, file_path: str) -> str:
        from docx import Document

        doc = Document(file_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _smart_chunk(self, text: str, doc_type: str) -> list[dict]:
        """智能分块"""
        if doc_type == "script":
            return self._chunk_script(text)
        return self._chunk_novel(text)

    def _chunk_script(self, text: str) -> list[dict]:
        """剧本分块：按场景分割"""
        chunks: list[dict] = []
        # Keep this pattern entirely non-capturing. The previous nested
        # captures made re.split insert None values for ordinary scene input.
        scene_pattern = r"(?:\[场景\s*\d+\]|SCENE\s*\d+|第[一二三四五六七八九十\d]+幕)"
        markers = list(re.finditer(scene_pattern, text or "", flags=re.IGNORECASE))
        segments: list[str] = []
        if markers:
            if text[: markers[0].start()].strip():
                segments.append(text[: markers[0].start()])
            for index, marker in enumerate(markers):
                end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
                segments.append(text[marker.start() : end])
        else:
            segments = [text or ""]

        for segment in segments:
            clean = segment.strip()
            if not clean:
                continue
            chunks.append(
                {
                    "text": clean,
                    "type": "scene",
                    "characters": self._extract_characters(clean),
                }
            )
        return chunks

    def _chunk_novel(self, text: str) -> list[dict]:
        """小说分块：滑动窗口"""
        chunks = []
        paragraphs = text.split("\n\n")
        window_size = 3
        overlap = 1

        for i in range(0, len(paragraphs), window_size - overlap):
            window = paragraphs[i : i + window_size]
            chunk_text = "\n\n".join(window)

            if len(chunk_text.strip()) < 50:
                continue

            chunk_type = self._classify_chunk(chunk_text)

            chunks.append(
                {
                    "text": chunk_text.strip(),
                    "type": chunk_type,
                    "chapter": self._detect_chapter(chunk_text),
                    "characters": self._extract_characters(chunk_text),
                    "location": self._extract_location(chunk_text),
                }
            )

        return chunks

    def _classify_chunk(self, text: str) -> str:
        """分类 chunk 类型"""
        dialogue_ratio = len(re.findall(r'[""「」]', text)) / max(len(text), 1)
        if dialogue_ratio > 0.05:
            return "dialogue"

        scene_keywords = ["房间", "教室", "街道", "庭院", "天空", "阳光", "灯光", "公园"]
        if any(kw in text for kw in scene_keywords):
            return "scene"

        return "narrative"

    def _extract_characters(self, text: str) -> list[str]:
        """提取文本中的人名（简单启发式）"""
        # 匹配中文人名模式（2-4个汉字的名词，后面跟动词或对话标记）
        name_pattern = r"(?:^|[，。！？\s])([A-Z]?[一-龥]{2,4})(?:说|道|笑|看|走|想|喊|叫)"
        matches = re.findall(name_pattern, text)
        return list(set(matches))[:10]  # 最多取 10 个

    def _detect_chapter(self, text: str) -> str:
        """检测章节标记"""
        chapter_pattern = r"第[一二三四五六七八九十\d]+[章回节幕]"
        match = re.search(chapter_pattern, text)
        return match.group() if match else ""

    def _extract_location(self, text: str) -> str:
        """提取场景地点"""
        location_keywords = ["在", "来到", "走进", "走出"]
        for kw in location_keywords:
            pattern = f"{kw}([\\u4e00-\\u9fa5]{{2,10}}(?:里|中|内|外|上|下))"
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return ""
