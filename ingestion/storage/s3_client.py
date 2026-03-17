# ingestion/storage/s3_client.py

import boto3
import tempfile
from pathlib import Path
from botocore.exceptions import ClientError

# S3 key convention: raw/{ticker}/{year}/{filename}
# This makes it easy to list all documents for a company or year
S3_PREFIX_TEMPLATE = "raw/{ticker}/{year}/{filename}"


class S3DocumentStore:
    """
    Wrapper around S3 for storing and retrieving financial PDF documents.

    Key design decisions:
    - Documents are stored with a structured key: raw/AAPL/2025/filename.pdf
      This makes it trivial to list all docs for a ticker or year
    - Downloads use temp files so we never need to persist to disk on the server
      This is critical for stateless ECS containers
    - All operations are logged for observability

    S3 bucket structure:
        finsight-rag-documents/
        └── raw/
            ├── AAPL/
            │   ├── 2025/
            │   │   └── AAPL_10K_2025.pdf
            │   └── 2024/
            │       └── AAPL_10K_2024.pdf
            └── MSFT/
                └── 2025/
                    └── MSFT_10K_2025.pdf
    """

    def __init__(self, bucket_name: str = "finsight-rag-documents"):
        self.bucket = bucket_name
        self.client = boto3.client("s3")
        print(f"☁️  S3 store initialized: s3://{bucket_name}")

    def upload(self, local_path: str, ticker: str, year: int) -> str:
        """
        Uploads a PDF to S3 and returns the S3 key.
        Safe to call multiple times — S3 versioning preserves old copies.
        """
        local_path = Path(local_path)
        s3_key = S3_PREFIX_TEMPLATE.format(
            ticker=ticker,
            year=year,
            filename=local_path.name,
        )

        print(f"⬆️  Uploading {local_path.name} to s3://{self.bucket}/{s3_key}")

        self.client.upload_file(
            str(local_path),
            self.bucket,
            s3_key,
            ExtraArgs={
                "Metadata": {
                    "ticker": ticker,
                    "year": str(year),
                    "source": "finsight-rag",
                }
            }
        )

        print(f"✅ Upload complete: s3://{self.bucket}/{s3_key}")
        return s3_key

    def download_to_temp(self, ticker: str, year: int, filename: str) -> str:
        """
        Downloads a PDF from S3 to a temporary file.
        Returns the temp file path for the parser to use.

        Why temp files?
        ECS containers are stateless — we don't want to persist
        files to the container filesystem between requests.
        tempfile handles cleanup automatically.
        """
        s3_key = S3_PREFIX_TEMPLATE.format(
            ticker=ticker,
            year=year,
            filename=filename,
        )

        print(f"⬇️  Downloading s3://{self.bucket}/{s3_key}")

        # Create a named temp file that persists until we delete it
        tmp = tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        )

        try:
            self.client.download_fileobj(
                self.bucket,
                s3_key,
                tmp,
            )
            tmp.flush()
            print(f"✅ Downloaded to temp: {tmp.name}")
            return tmp.name

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                raise FileNotFoundError(
                    f"Document not found in S3: s3://{self.bucket}/{s3_key}"
                )
            raise

    def list_documents(self, ticker: str = None, year: int = None) -> list[dict]:
        """
        Lists documents in S3, optionally filtered by ticker and/or year.
        Useful for the /health endpoint and admin tooling.
        """
        prefix = "raw/"
        if ticker:
            prefix += f"{ticker}/"
        if ticker and year:
            prefix += f"{year}/"

        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=prefix,
        )

        documents = []
        for obj in response.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) >= 4:
                documents.append({
                    "s3_key": obj["Key"],
                    "ticker": parts[1],
                    "year": parts[2],
                    "filename": parts[3],
                    "size_mb": round(obj["Size"] / 1_000_000, 2),
                    "last_modified": obj["LastModified"].isoformat(),
                })

        return documents

    def document_exists(self, ticker: str, year: int, filename: str) -> bool:
        """Check if a document exists in S3 without downloading it."""
        s3_key = S3_PREFIX_TEMPLATE.format(
            ticker=ticker,
            year=year,
            filename=filename,
        )
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False