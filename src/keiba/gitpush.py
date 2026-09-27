"""成果物を GitHub に保存する

各ノートブックは自分の成果物を必ず GitHub に置く。次のノートブックは
それを取りに行く。これにより、どのノートブックも別々のランタイムで
実行できる。

旧プロジェクトはこれが徹底されておらず、C01 が取得したHTMLを保存して
いなかったため、C02 を別タブで開くと C01 の成果が見えないという
暗黙の前提ができていた（結果として 432 レース分のHTMLが失われた）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

from . import config


def _run(args: Sequence[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True, **kw)


def get_token() -> Optional[str]:
    """Colab のシークレット、環境変数、手入力の順に探す。"""
    try:
        from google.colab import userdata  # type: ignore
        token = userdata.get('GITHUB_TOKEN')
        if token:
            print('🔑 Colabシークレットからトークンを取得しました')
            return token
    except Exception:
        pass

    import os
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        print('🔑 環境変数 GITHUB_TOKEN からトークンを取得しました')
        return token

    try:
        import getpass
        return getpass.getpass('GitHub Personal Access Token: ') or None
    except Exception:
        return None


def push(paths: List[str], message: str,
         repo_root: Optional[Path] = None,
         token: Optional[str] = None) -> bool:
    """指定パスをコミットして push する。

    Args:
        paths: git add する相対パス
        message: コミットメッセージ
    Returns:
        push まで成功したら True
    """
    root = repo_root or config.PROJECT_ROOT

    if not (root / '.git').exists():
        print(f'⚠️ {root} は git リポジトリではありません。保存をスキップします')
        return False

    status = _run(['git', 'status', '--porcelain', *paths], cwd=root)
    if not status.stdout.strip():
        print('ℹ️ 変更なし')
        return True

    n = len(status.stdout.strip().splitlines())
    print(f'📤 {n:,} ファイルの変更を保存します...')

    token = token or get_token()
    if token:
        url = (f'https://oauth2:{token}@github.com/'
               f'{config.GITHUB_OWNER}/{config.GITHUB_REPO}.git')
        _run(['git', 'remote', 'set-url', 'origin', url], cwd=root)
    else:
        print('⚠️ トークンが無いため、既存の認証設定で push を試みます')

    _run(['git', 'config', 'user.email', 'colab-bot@example.com'], cwd=root)
    _run(['git', 'config', 'user.name', 'Colab Bot'], cwd=root)

    add = _run(['git', 'add', *paths], cwd=root)
    if add.returncode != 0:
        print(f'❌ git add 失敗: {add.stderr}')
        return False

    commit = _run(['git', 'commit', '-m', message], cwd=root)
    if commit.returncode != 0 and 'nothing to commit' not in commit.stdout + commit.stderr:
        print(f'❌ commit 失敗: {commit.stderr or commit.stdout}')
        return False

    # 他のノートブックの push と競合しないよう、先にリモートを取り込む
    pull = _run(['git', 'pull', '--rebase'], cwd=root)
    if pull.returncode != 0:
        print(f'⚠️ pull --rebase 失敗: {pull.stderr.strip()[:200]}')

    result = _run(['git', 'push'], cwd=root)
    if result.returncode == 0:
        print('✅ プッシュ完了')
        return True
    print(f'❌ push 失敗: {result.stderr.strip()[:300]}')
    return False
