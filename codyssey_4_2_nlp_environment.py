#!/usr/bin/env python3
"""Codyssey 4-2 sentiment NLP prerequisites: sudo-free, two-phase setup.

Place at 4-2/codyssey_4_2_nlp_environment.py (Python >= 3.10). Every artifact created
by this program except the Python virtual environment lives below 4-2/data/.
The virtual environment is 4-2/.venv, matching the Jupyter kernel. Only a
pre-existing host Python and optionally a system JDK are used from elsewhere.

Examples:
    python3 codyssey_4_2_nlp_environment.py                # install ALL, then check ALL
    python3 codyssey_4_2_nlp_environment.py --dry-run      # explain without writing
    python3 codyssey_4_2_nlp_environment.py --install-only
    python3 codyssey_4_2_nlp_environment.py --check        # --check-only alias
    python3 codyssey_4_2_nlp_environment.py --brief-env
    python3 codyssey_4_2_nlp_environment.py --java local --jdk-version 25

A check is intentionally separate from installation. No target's install()
method invokes its own check(); Master runs check_all() after install_all().
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
import csv
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import venv


KNU_URL = (
    "https://raw.githubusercontent.com/park1200656/knu_senti_dict/"
    "master/SentiWord_info.json"
)
NSMC_FILES = {
    "ratings_train.txt": (150_000, "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"),
    "ratings_test.txt": (50_000, "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"),
}
NLTK_PACKAGES = ("vader_lexicon", "punkt_tab", "wordnet", "movie_reviews")
PACKAGE_VERSIONS = {"konlpy": "0.6.0", "JPype1": "1.6.0", "nltk": "3.9.2"}
NOTEBOOK_PACKAGES = {"pandas": "2.3.3", "ipykernel": "6.30.1"}


def log(message: str) -> None:
    print(f"[setup] {message}", flush=True)


def execute(*cmd: str, env: dict[str, str] | None = None,
            cwd: Path | None = None, capture: bool = False,
            check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(map(str, cmd)), text=True, env=env, cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=check,
    )


def platform_profile() -> str:
    machine = platform.machine().lower()
    if sys.platform == "linux" and machine in ("x86_64", "amd64"):
        return "linux-x64"
    if sys.platform == "darwin" and machine in ("x86_64", "amd64"):
        return "mac-intel"
    raise RuntimeError(
        f"Unsupported platform: {sys.platform}/{machine}; "
        "this script targets Linux x86_64 and Intel macOS only"
    )


def digest_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def download(url: str, destination: Path, *, expected_sha256: str | None = None,
             validator=None) -> None:
    """Skip valid files; stage downloads in the same directory before replace."""
    if destination.is_file():
        hash_ok = not expected_sha256 or digest_sha256(destination).lower() == expected_sha256.lower()
        content_ok = validator is None or validator(destination)
        if hash_ok and content_ok:
            log(f"reuse {destination.name}")
            return
        log(f"replace invalid {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".partial-", delete=False
    ) as f:
        tmp_path = Path(f.name)
    try:
        log(f"download {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "Codyssey-NLP-Setup/2.0"})
        with urllib.request.urlopen(request, timeout=90) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if expected_sha256 and digest_sha256(tmp_path).lower() != expected_sha256.lower():
            raise RuntimeError(f"SHA-256 mismatch: {destination.name}")
        if validator is not None and not validator(tmp_path):
            raise RuntimeError(f"Invalid downloaded data: {destination.name}")
        tmp_path.replace(destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def http_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "Codyssey-NLP-Setup/2.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Handle Python 3.10's missing tar extraction filters safely."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        if "filter" in inspect.signature(tar.extractall).parameters:
            tar.extractall(destination, filter="data")
            return
        # On Python 3.10 validate members before extraction, including links.
        import posixpath
        for member in tar.getmembers():
            name = posixpath.normpath(member.name)
            if name == ".." or name.startswith("../") or name.startswith("/"):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                raise RuntimeError(f"Unsupported archive member: {member.name}")
            if member.issym() or member.islnk():
                base = posixpath.dirname(name) if member.issym() else ""
                target = posixpath.normpath(posixpath.join(base, member.linkname))
                if member.linkname.startswith("/") or target == ".." or target.startswith("../"):
                    raise RuntimeError(f"Unsafe archive link: {member.name}")
        tar.extractall(destination)


def java_major(home: Path) -> int | None:
    java_bin = home / "bin" / "java"
    if not java_bin.is_file():
        return None
    try:
        output = execute(java_bin, "-version", capture=True, check=False)
    except OSError:
        return None
    if output.returncode != 0:
        return None
    match = re.search(r'(?:openjdk|java) version "?(\d+)(?:[.\-+"\s]|$)',
                      output.stdout + output.stderr)
    if not match:
        return None
    major = int(match.group(1))
    return 8 if major == 1 else major


def find_system_jdk(version: int) -> Path | None:
    homes: list[Path] = []
    if os.environ.get("JAVA_HOME"):
        homes.append(Path(os.environ["JAVA_HOME"]))
    if sys.platform == "darwin" and Path("/usr/libexec/java_home").exists():
        result = execute("/usr/libexec/java_home", "-v", str(version), capture=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            homes.append(Path(result.stdout.strip()))
    binary = shutil.which("java")
    if binary:
        homes.append(Path(binary).resolve().parent.parent)
    for home in homes:
        if java_major(home) == version and (home / "bin" / "javac").is_file():
            return home.resolve()
    return None


@dataclass
class Context:
    root: Path
    profile: str
    java_mode: str = "auto"
    jdk_version: int = 25
    java_home: Path | None = None
    java_local: bool | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def venv_dir(self) -> Path:
        return self.root / ".venv"

    @property
    def python(self) -> Path:
        return self.venv_dir / "bin" / "python"

    @property
    def nltk_dir(self) -> Path:
        return self.data / "nltk_data"

    @property
    def knu_file(self) -> Path:
        return self.data / "lexicons" / "ko" / "SentiWord_info.json"

    @property
    def nsmc_dir(self) -> Path:
        return self.data / "datasets" / "nsmc"

    @property
    def local_jdk_dir(self) -> Path:
        return self.data / f"jdk-{self.jdk_version}"

    @property
    def local_jdk_home(self) -> Path:
        return (self.local_jdk_dir / "Contents" / "Home"
                if self.profile == "mac-intel" else self.local_jdk_dir)

    def child_env(self) -> dict[str, str]:
        """Child processes receive needed variables without sudo or shell edits."""
        env = dict(os.environ)
        env.update({
            "PIP_CACHE_DIR": str(self.data / ".cache" / "pip"),
            "TMPDIR": str(self.data / ".tmp"),
            "NLTK_DATA": str(self.nltk_dir),
            "PYTHONUTF8": "1",
        })
        if self.java_home is not None:
            env["JAVA_HOME"] = str(self.java_home)
            env["PATH"] = str(self.java_home / "bin") + os.pathsep + env.get("PATH", "")
        return env

    def run_python(self, *args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
        return execute(self.python, *args, env=self.child_env(), cwd=self.root, capture=capture)


class SetupTarget(ABC):
    name: str

    def __init__(self, ctx: Context):
        self.ctx = ctx

    @abstractmethod
    def install(self) -> None:
        """Install/download only; verification belongs to check()."""

    @abstractmethod
    def check(self) -> None:
        """Read-only verification; raise on failure."""

    @abstractmethod
    def brief_env(self) -> str:
        """Explain locations, dependencies and environment variables."""


class Java(SetupTarget):
    name = "Java JDK"

    def _resolve_existing(self) -> tuple[Path, bool] | None:
        system = find_system_jdk(self.ctx.jdk_version)
        local = self.ctx.local_jdk_home
        local_ok = java_major(local) == self.ctx.jdk_version and (local / "bin" / "javac").is_file()
        if self.ctx.java_mode == "system":
            return (system, False) if system else None
        if self.ctx.java_mode == "local":
            return (local, True) if local_ok else None
        # Linux auto prefers a matching preinstalled JDK; Intel Mac auto prefers local.
        if self.ctx.profile == "linux-x64":
            if system:
                return (system, False)
            return (local, True) if local_ok else None
        if local_ok:
            return local, True
        return None

    def _set_java(self, home: Path, local: bool) -> None:
        self.ctx.java_home = home
        self.ctx.java_local = local

    def install(self) -> None:
        existing = self._resolve_existing()
        if existing:
            self._set_java(*existing)
            log(f"reuse JDK {self.ctx.jdk_version}: {self.ctx.java_home}")
            return
        if self.ctx.java_mode == "system":
            raise RuntimeError(f"System JDK {self.ctx.jdk_version} not found")
        if self.ctx.local_jdk_dir.exists():
            raise RuntimeError(f"Incomplete local JDK: {self.ctx.local_jdk_dir}; remove this folder and retry")

        os_name = "mac" if self.ctx.profile == "mac-intel" else "linux"
        query = urllib.parse.urlencode({
            "architecture": "x64", "image_type": "jdk", "os": os_name, "vendor": "eclipse",
        })
        endpoint = f"https://api.adoptium.net/v3/assets/latest/{self.ctx.jdk_version}/hotspot?{query}"
        assets = http_json(endpoint)
        if not isinstance(assets, list) or not assets:
            raise RuntimeError(f"Temurin JDK {self.ctx.jdk_version} unavailable for {os_name}/x64")
        package = assets[0]["binary"]["package"]
        url, checksum = package["link"], package["checksum"]
        archive = self.ctx.data / "downloads" / Path(package["name"]).name
        download(url, archive, expected_sha256=checksum)

        tmp_dir = self.ctx.data / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="jdk-", dir=tmp_dir) as scratch:
            folder = Path(scratch)
            safe_extract_tar(archive, folder)
            candidates = [p for p in folder.iterdir() if p.is_dir() and (
                (p / "Contents" / "Home" / "bin" / "java").is_file()
                if self.ctx.profile == "mac-intel"
                else (p / "bin" / "java").is_file()
            )]
            if len(candidates) != 1:
                raise RuntimeError("Unexpected JDK archive directory layout")
            shutil.move(str(candidates[0]), str(self.ctx.local_jdk_dir))
        # Do not perform the Java smoke test here; Master.check_all() does it.
        self._set_java(self.ctx.local_jdk_home, True)

    def check(self) -> None:
        existing = self._resolve_existing()
        if existing is None:
            raise RuntimeError(f"JDK {self.ctx.jdk_version} missing or not executable")
        home, is_local = existing
        if not (home / "bin" / "javac").is_file():
            raise RuntimeError(f"JDK compiler missing: {home}")
        self._set_java(home, is_local)
        log(f"JAVA_HOME={home} (JDK {java_major(home)})")

    def brief_env(self) -> str:
        return (
            f"JDK {self.ctx.jdk_version}; strategy={self.ctx.java_mode}; "
            f"Intel Mac: data/jdk-{self.ctx.jdk_version} (no sudo), "
            "Linux: reuse matching system JDK when available, otherwise local. "
            "JAVA_HOME/PATH are applied to child processes automatically."
        )


class PythonVirtualEnv(SetupTarget):
    name = "Python virtual environment"

    def install(self) -> None:
        self.ctx.venv_dir.parent.mkdir(parents=True, exist_ok=True)
        py = self.ctx.python
        if py.is_file():
            log(f"reuse root virtualenv {py}")
            return
        if self.ctx.venv_dir.exists():
            raise RuntimeError(f"Incomplete Python environment: {self.ctx.venv_dir}; remove and retry")
        # Relocatable host Pythons (e.g. uv) find their runtime relative to the executable.
        venv.EnvBuilder(with_pip=True, symlinks=True).create(self.ctx.venv_dir)

    def check(self) -> None:
        py = self.ctx.python
        if not py.is_file():
            raise RuntimeError(f"Missing virtualenv interpreter: {py}")
        out = self.ctx.run_python(
            "-c", "import sys; print(sys.prefix != sys.base_prefix); print('%d.%d' % sys.version_info[:2])",
            capture=True,
        ).stdout.splitlines()
        current = f"{sys.version_info.major}.{sys.version_info.minor}"
        if len(out) != 2 or out[0].strip() != "True" or out[1].strip() != current:
            raise RuntimeError(f"Invalid virtualenv or Python version mismatch: {out!r}, expected {current}")
        log(f"venv Python {current}: {py}")

    def brief_env(self) -> str:
        return "Uses existing 4-2/.venv or creates it with host Python >=3.10; no global pip or sudo."


class KoNLPy(SetupTarget):
    name = "KoNLPy / Okt"

    def install(self) -> None:
        self.ctx.run_python(
            "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
            f"konlpy=={PACKAGE_VERSIONS['konlpy']}",
            f"JPype1=={PACKAGE_VERSIONS['JPype1']}",
        )

    def check(self) -> None:
        code = '''
from importlib.metadata import version
from konlpy.tag import Okt
assert version('konlpy') == '0.6.0'
assert version('JPype1') == '1.6.0'
result = Okt().pos('영화가 정말 재미있어요', norm=True, stem=True)
assert result and any(tag.startswith('N') for _, tag in result)
print('Okt:', result)
'''
        self.ctx.run_python("-c", code)

    def brief_env(self) -> str:
        return "KoNLPy 0.6.0 + JPype1 1.6.0 in root .venv; Okt requires JAVA_HOME."


class NotebookRuntime(SetupTarget):
    name = "Assignment notebook kernel"

    def install(self) -> None:
        self.ctx.run_python(
            "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
            *(f"{name}=={version}" for name, version in NOTEBOOK_PACKAGES.items()),
        )

    def check(self) -> None:
        self.ctx.run_python("-c", """
import ipykernel
import pandas as pd
import numpy as np
from IPython.display import display
assert pd.Series(np.array([1, -1])).sum() == 0
print('Notebook kernel:', ipykernel.__version__, 'pandas:', pd.__version__)
""")

    def brief_env(self) -> str:
        return "Select .venv/bin/python as the notebook kernel after sourcing data/activate.sh."


class NLTK(SetupTarget):
    name = "NLTK / VADER / English corpora"

    def install(self) -> None:
        self.ctx.run_python(
            "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
            f"nltk=={PACKAGE_VERSIONS['nltk']}",
        )
        self.ctx.nltk_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.run_python(
            "-m", "nltk.downloader", "-d", str(self.ctx.nltk_dir), *NLTK_PACKAGES
        )

    def check(self) -> None:
        code = '''
from importlib.metadata import version
from nltk.corpus import movie_reviews, wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
assert version('nltk') == '3.9.2'
assert len(movie_reviews.fileids('pos')) == 1000
assert len(movie_reviews.fileids('neg')) == 1000
assert wordnet.synsets('movie')
assert word_tokenize('This movie was very good.')
lex = SentimentIntensityAnalyzer().lexicon
assert len(lex) >= 1000 and 'good' in lex
print('Movie Reviews: 1000 positive / 1000 negative; VADER:', len(lex), 'entries')
'''
        self.ctx.run_python("-c", code)

    def brief_env(self) -> str:
        return (
            "NLTK 3.9.2 in root .venv; vader_lexicon, punkt_tab, wordnet, movie_reviews "
            "in data/nltk_data. NLTK_DATA is automatically set for child processes."
        )


def valid_knu(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8-sig") as source:
            rows = json.load(source)
        return (
            isinstance(rows, list) and len(rows) >= 200
            and all(isinstance(row, dict) and {"word", "polarity"} <= row.keys()
                    for row in rows)
        )
    except (OSError, ValueError, UnicodeError, TypeError):
        return False


class KNULexicon(SetupTarget):
    name = "KNU Korean sentiment lexicon"

    def install(self) -> None:
        download(KNU_URL, self.ctx.knu_file, validator=valid_knu)

    def check(self) -> None:
        if not valid_knu(self.ctx.knu_file):
            raise RuntimeError(f"KNU lexicon missing/invalid: {self.ctx.knu_file}")
        with self.ctx.knu_file.open(encoding="utf-8-sig") as source:
            rows = json.load(source)
        log(f"KNU entries: {len(rows)}; sha256={digest_sha256(self.ctx.knu_file)}")

    def brief_env(self) -> str:
        return f"Original KNU JSON at data/lexicons/ko/SentiWord_info.json; source: {KNU_URL}"


def valid_nsmc(path: Path, expected_count: int) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if reader.fieldnames != ["id", "document", "label"]:
                return False
            count = 0
            for row in reader:
                if row["label"] not in ("0", "1"):
                    return False
                count += 1
            return count == expected_count
    except (OSError, UnicodeError, ValueError, TypeError):
        return False


class NSMCDataset(SetupTarget):
    name = "NSMC Korean review dataset"

    def install(self) -> None:
        for name, (count, url) in NSMC_FILES.items():
            download(url, self.ctx.nsmc_dir / name,
                     validator=lambda file, n=count: valid_nsmc(file, n))

    def check(self) -> None:
        for name, (count, _) in NSMC_FILES.items():
            file = self.ctx.nsmc_dir / name
            if not valid_nsmc(file, count):
                raise RuntimeError(f"NSMC missing/invalid: {file}")
            log(f"NSMC {name}: {count} records; sha256={digest_sha256(file)}")

    def brief_env(self) -> str:
        return "NSMC train/test TSV under data/datasets/nsmc; original train=150k, test=50k."


class ShellEnvironment(SetupTarget):
    name = "Shell activation script"

    def _content(self) -> str:
        if self.ctx.java_home is None:
            raise RuntimeError("Java location unknown; Java.install() must execute first")
        if self.ctx.java_local:
            java_expr = f'"$_CODYSSEY_DATA/jdk-{self.ctx.jdk_version}'
            if self.ctx.profile == "mac-intel":
                java_expr += '/Contents/Home'
            java_expr += '"'
        else:
            java_expr = json.dumps(str(self.ctx.java_home))
        return (
            '#!/usr/bin/env bash\n'
            '# Run from bash or zsh: source /path/to/4-2/data/activate.sh\n'
            'if [ -n "${ZSH_VERSION:-}" ]; then\n'
            '  _CODYSSEY_SOURCE="${(%):-%N}"\n'
            'else\n'
            '  _CODYSSEY_SOURCE="${BASH_SOURCE[0]}"\n'
            'fi\n'
            '_CODYSSEY_DATA="$(cd "$(dirname "$_CODYSSEY_SOURCE")" && pwd)"\n'
            'unset _CODYSSEY_SOURCE\n'
            'export NLTK_DATA="$_CODYSSEY_DATA/nltk_data"\n'
            f'export JAVA_HOME={java_expr}\n'
            'export PATH="$JAVA_HOME/bin:$PATH"\n'
            'source "$_CODYSSEY_DATA/../.venv/bin/activate"\n'
            'unset _CODYSSEY_DATA\n'
        )

    def install(self) -> None:
        script = self.ctx.data / "activate.sh"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(self._content(), encoding="utf-8")

    def check(self) -> None:
        script = self.ctx.data / "activate.sh"
        if not script.is_file():
            raise RuntimeError(f"Activation script missing: {script}")
        if script.read_text(encoding="utf-8") != self._content():
            raise RuntimeError("Activation script outdated; rerun install")
        execute("bash", "-n", str(script), cwd=self.ctx.root)

    def brief_env(self) -> str:
        return (
            "In commands launched by setup, JAVA_HOME/PATH/NLTK_DATA are automatic. "
            "For an interactive shell run: source data/activate.sh (not possible to "
            "permanently alter parent shell from Python)."
        )


class Master:
    """Keep install and check phases separate, preserving registration order."""

    def __init__(self) -> None:
        self.targets: list[SetupTarget] = []
        self.install_results: dict[str, str] = {}
        self.check_results: dict[str, str] = {}

    def add(self, target: SetupTarget) -> "Master":
        if any(existing.name == target.name for existing in self.targets):
            raise ValueError(f"Duplicate target: {target.name}")
        self.targets.append(target)
        return self

    def install_all(self) -> bool:
        log("========== PHASE 1: INSTALL ALL ==========")
        for target in self.targets:
            log(f"INSTALL [{target.name}]")
            try:
                target.install()
                self.install_results[target.name] = "OK"
            except Exception as exc:
                self.install_results[target.name] = f"FAILED: {exc}"
                log(f"INSTALL FAILED [{target.name}]: {exc}")
        return all(value == "OK" for value in self.install_results.values())

    def check_all(self) -> bool:
        log("========== PHASE 2: CHECK ALL ==========")
        for target in self.targets:
            log(f"CHECK   [{target.name}]")
            try:
                target.check()
                self.check_results[target.name] = "OK"
            except Exception as exc:
                self.check_results[target.name] = f"FAILED: {exc}"
                log(f"CHECK FAILED [{target.name}]: {exc}")
        return all(value == "OK" for value in self.check_results.values())

    def brief_env_all(self) -> None:
        log("========== ENVIRONMENT / LOCATIONS ==========")
        for target in self.targets:
            log(f"{target.name}: {target.brief_env()}")

    def report(self) -> bool:
        log("========== SUMMARY ==========")
        for target in self.targets:
            install = self.install_results.get(target.name, "not run")
            check = self.check_results.get(target.name, "not run")
            log(f"{target.name}: install={install}; check={check}")
        return all(v == "OK" for v in (*self.install_results.values(), *self.check_results.values()))


def build_master(ctx: Context) -> Master:
    """Registration order reflects prerequisites; no checks happen here."""
    return (Master()
            .add(Java(ctx))
            .add(PythonVirtualEnv(ctx))
            .add(KoNLPy(ctx))
            .add(NotebookRuntime(ctx))
            .add(NLTK(ctx))
            .add(KNULexicon(ctx))
            .add(NSMCDataset(ctx))
            .add(ShellEnvironment(ctx)))


def write_manifest(ctx: Context) -> None:
    """Record reproducible metadata only after successful full verification."""
    manifest = {
        "platform": ctx.profile,
        "python": sys.version.split()[0],
        "venv_path": str(ctx.venv_dir),
        "java_major": ctx.jdk_version,
        "java_mode": "local" if ctx.java_local else "system",
        "java_home": str(ctx.java_home),
        "python_packages": {**PACKAGE_VERSIONS, **NOTEBOOK_PACKAGES},
        "nltk_packages": list(NLTK_PACKAGES),
        "sources": {"KNU": KNU_URL, **{k: v[1] for k, v in NSMC_FILES.items()}},
        "sha256": {
            "KNU": digest_sha256(ctx.knu_file),
            **{name: digest_sha256(ctx.nsmc_dir / name) for name in NSMC_FILES},
        },
    }
    (ctx.data / "setup_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    frozen = ctx.run_python("-m", "pip", "freeze", capture=True).stdout
    (ctx.data / "pip-freeze.txt").write_text(frozen, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--java", choices=("auto", "local", "system"), default="auto")
    parser.add_argument("--jdk-version", type=int, default=25)
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--install-only", action="store_true", help="Install every target, skip checks")
    commands.add_argument("--check", "--check-only", dest="check_only", action="store_true",
                          help="Check every target without installing or changing files")
    commands.add_argument("--brief-env", action="store_true", help="Explain paths and environment variables")
    commands.add_argument("--dry-run", action="store_true", help="Show plan without changing files")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 10):
        parser.error("Python >= 3.10 required")
    if args.jdk_version < 17:
        parser.error("Use JDK 17 or later")
    ctx = Context(root=Path(__file__).resolve().parent,
                  profile=platform_profile(), java_mode=args.java,
                  jdk_version=args.jdk_version)
    master = build_master(ctx)
    log(f"Platform: {ctx.profile} | data root: {ctx.data}")
    if args.brief_env or args.dry_run:
        master.brief_env_all()
        if args.dry_run:
            log("PLAN: install_all() followed by check_all(); no changes made")
        return 0

    if not args.check_only:
        # Set up pip/temp directories in data/ before initiating installs.
        for path in (ctx.data / ".cache" / "pip", ctx.data / ".tmp"):
            path.mkdir(parents=True, exist_ok=True)
        master.install_all()

    if not args.install_only:
        master.check_all()

    ok = master.report()
    if ok and not args.install_only and not args.check_only:
        write_manifest(ctx)
        log("Full setup complete. For interactive use: source data/activate.sh")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, urllib.error.URLError, subprocess.CalledProcessError) as exc:
        print(f"[setup] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
