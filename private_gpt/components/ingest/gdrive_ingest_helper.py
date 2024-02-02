import logging
import io
from pathlib import Path
import tempfile
from typing import Any

from injector import inject, singleton
from llama_index import Document

from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GDriveIngestHelper():

    @staticmethod
    def _raise_import_error():
        raise ImportError(
            "You must run "
            "`pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib` "
            "to ingest content from Google Drive."
        )

    @staticmethod
    def _get_creds():
        try:
            from google.auth import default
            from google.oauth2 import service_account
        except ImportError:
            self._raise_import_error()

        service_account_key = settings().gdrive.service_account_key
        if service_account_key:
            return service_account.Credentials.from_service_account_file(
                str(service_account_key),
                scopes=SCOPES
            )

        creds, _ = default(scopes=SCOPES)
        return creds

    @classmethod
    def _get_file_content(cls, service, file_id) -> bytes:
        try:
            from googleapiclient.errors import HttpError
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError:
            cls._raise_import_error()

        try:
            request = service.files().get_media(fileId=file_id)
            content_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(content_buffer, request)
            done = False
            while done is False:
              _, done = downloader.next_chunk()

        except HttpError as error:
          logger.error("Failed to download the file %s", file_id, error)
          return None

        return content_buffer.getvalue()

    @classmethod
    def _handle_non_gdrive_file(cls, service, file) -> list[Document]:
        content = cls._get_file_content(service, file['id'])
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                path_to_tmp.write_bytes(content)
                return IngestionHelper.transform_file_into_documents(file['name'], path_to_tmp)
            finally:
                tmp.close()
                path_to_tmp.unlink()

    @classmethod
    def _get_all_files(cls, service, folder_id):
        results = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents",
                pageSize=1000,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                fields="nextPageToken, files(id, name, mimeType, parents, trashed)",
            )
            .execute()
        )
        files = results.get("files", [])
        returns = []
        for file in files:
            if file["mimeType"] == "application/vnd.google-apps.folder":
                if settings().gdrive.recursive:
                    returns.extend(cls._fetch_files_recursive(service, file["id"]))
            else:
                returns.append(file)

        return returns

    @classmethod
    def _handle_gdoc(cls, creds, document_id: str) -> Document:
        try:
            import googleapiclient.discovery as discovery
        except ImportError:
            cls._raise_import_error()

        docs_service = discovery.build("docs", "v1", credentials=creds)
        doc = docs_service.documents().get(documentId=document_id).execute()
        processed_doc = cls._read_structural_elements(doc.get("body").get("content"))
        return Document(text=processed_doc, metadata={"document_id": document_id})

    @staticmethod
    def _read_paragraph_element(element: Any) -> Any:
        """Return the text in the given ParagraphElement.

        Args:
            element: a ParagraphElement from a Google Doc.
        """
        text_run = element.get("textRun")
        if not text_run:
            return ""
        return text_run.get("content")

    @classmethod
    def _read_structural_elements(cls, elements: list[Any]) -> Any:
        """Recurse through a list of Structural Elements.

        Read a document's text where text may be in nested elements.

        Args:
            elements: a list of Structural Elements.
        """
        text = ""
        for value in elements:
            if "paragraph" in value:
                elements = value.get("paragraph").get("elements")
                for elem in elements:
                    text += cls._read_paragraph_element(elem)
            elif "table" in value:
                # The text in table cells are in nested Structural Elements
                # and tables may be nested.
                table = value.get("table")
                for row in table.get("tableRows"):
                    cells = row.get("tableCells")
                    for cell in cells:
                        text += cls._read_structural_elements(cell.get("content"))
            elif "tableOfContents" in value:
                # The text in the TOC is also in a Structural Element.
                toc = value.get("tableOfContents")
                text += cls._read_structural_elements(toc.get("content"))
        return text

    @classmethod
    def transform_into_documents(cls, folder_id: str) -> list[Document]:
        try:
            from googleapiclient.discovery import build
        except ImportError:
            cls._raise_import_error()

        creds = cls._get_creds()
        service = build("drive", "v3", credentials=creds)
        files = cls._get_all_files(service, folder_id)

        docs = []
        for file in files:
            if file["trashed"] and not settings().gdrive.load_trashed_files:
                continue
            match file["mimeType"]:
                case "application/vnd.google-apps.document":
                    docs.append(cls._handle_gdoc(creds, file['id']))
                # case "application/vnd.google-apps.spreadsheet":
                #     pass
                case _:
                    if "google-apps" in file["mimeType"]:
                        logger.debug("Google Drive file type %s is currently not supported", file["mimeType"])
                        continue
                    docs.extend(cls._handle_non_gdrive_file(service, file))
        return docs
