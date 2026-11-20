"""Release packaging script for PoseGuard AI v1.0.0-beta.
Builds the official source distribution ZIP, generates SHA-256 checksums,
and outputs a complete release manifest.
"""

import os
import sys
import zipfile
import hashlib
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT_DIR / "dist"
ARCHIVE_NAME = "poseguard-ai-1.0.0-beta.zip"
ARCHIVE_PATH = DIST_DIR / ARCHIVE_NAME
CHECKSUM_PATH = DIST_DIR / f"{ARCHIVE_NAME}.sha256"
MANIFEST_PATH = ROOT_DIR / "RELEASE_MANIFEST.json"

EXCLUDE_DIRS = {
    ".venv", "venv", "__pycache__", ".pytest_cache", ".git", ".github",
    "dist", "build", "poseguard.egg-info", "synthetic", "processed"
}

EXCLUDE_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".tmp", ".log", ".swp"
}

# Directories and files to explicitly include in the official release package
INCLUDED_ROOT_FILES = [
    "README.md", "LICENSE", "CHANGELOG.md", "CONTRIBUTING.md",
    "SECURITY.md", "PRIVACY.md", "CODE_OF_CONDUCT.md", "CITATION.cff",
    "pyproject.toml", "setup.py", "requirements.txt", "requirements-dev.txt"
]

INCLUDED_DIRS = [
    "poseguard",
    "models",
    Path("data") / "raw",
    "docs"
]

def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def build_release():
    print(f"=== Building PoseGuard AI Release Distribution ===")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
        
    manifest_entries = []
    
    with zipfile.ZipFile(ARCHIVE_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add individual root files
        for filename in INCLUDED_ROOT_FILES:
            src_file = ROOT_DIR / filename
            if src_file.exists():
                arcname = f"poseguard-ai-1.0.0-beta/{filename}"
                zipf.write(src_file, arcname)
                f_hash = compute_sha256(src_file)
                f_size = src_file.stat().st_size
                manifest_entries.append({
                    "path": arcname,
                    "size_bytes": f_size,
                    "sha256": f_hash
                })
                print(f"  + Added {arcname} ({f_size} bytes)")
                
        # 2. Add included directories
        for target_dir in INCLUDED_DIRS:
            abs_target = ROOT_DIR / target_dir
            if not abs_target.exists():
                print(f"  ! Warning: {target_dir} not found")
                continue
                
            for root, dirs, files in os.walk(abs_target):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in sorted(files):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in EXCLUDE_EXTENSIONS:
                        continue
                    full_p = Path(root) / file
                    rel_to_root = full_p.relative_to(ROOT_DIR)
                    arcname = f"poseguard-ai-1.0.0-beta/{rel_to_root.as_posix()}"
                    zipf.write(full_p, arcname)
                    f_hash = compute_sha256(full_p)
                    f_size = full_p.stat().st_size
                    manifest_entries.append({
                        "path": arcname,
                        "size_bytes": f_size,
                        "sha256": f_hash
                    })

    # Compute checksum of archive
    archive_sha256 = compute_sha256(ARCHIVE_PATH)
    archive_size = ARCHIVE_PATH.stat().st_size
    
    with open(CHECKSUM_PATH, "w") as f:
        f.write(f"{archive_sha256}  {ARCHIVE_NAME}\n")
        
    manifest = {
        "release": "PoseGuard AI v1.0.0-beta",
        "version": "1.0.0-beta",
        "date": "2026-11-19",
        "archive_filename": ARCHIVE_NAME,
        "archive_size_bytes": archive_size,
        "archive_size_mb": round(archive_size / (1024 * 1024), 3),
        "archive_sha256": archive_sha256,
        "total_packaged_files": len(manifest_entries),
        "model_checkpoint": {
            "name": "poseguard_v1.pt",
            "size_bytes": (ROOT_DIR / "models" / "poseguard_v1.pt").stat().st_size,
            "sha256": compute_sha256(ROOT_DIR / "models" / "poseguard_v1.pt"),
            "parameters": 214597
        },
        "files": manifest_entries
    }
    
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n" + "=" * 60)
    print("  RELEASE ARCHIVE CREATED SUCCESSFULLY")
    print(f"  Archive:  {ARCHIVE_PATH}")
    print(f"  Size:     {archive_size} bytes ({manifest['archive_size_mb']} MB)")
    print(f"  SHA-256:  {archive_sha256}")
    print(f"  Checksum: {CHECKSUM_PATH}")
    print(f"  Manifest: {MANIFEST_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    build_release()

