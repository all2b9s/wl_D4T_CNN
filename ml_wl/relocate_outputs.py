"""Relocate legacy rounds after all jobs using them have terminated."""
import json
import shutil
import sqlite3
from pathlib import Path

from ml_wl.config import SG_DIR


def relocate(sg=SG_DIR, work=Path('/work/hdd/bfmo/wenyinli')):
    legacy = sg / 'ml_wl' / 'rounds'
    destination = work / 'measurement' / 'ml_wl'
    if not legacy.exists():
        return
    rounds = sorted(legacy.iterdir())
    replacements = {}
    for rd in rounds:
        target = destination / rd.name
        logs = sg / 'output' / 'ml_wl' / rd.name
        replacements[str(rd / 'logs')] = str(logs)
        replacements[str(rd)] = str(target)
        replacements[str(work / 'ml_wl' / rd.name)] = str(target)
        if target.exists() or logs.exists():
            raise FileExistsError(f'Migration destination already exists: {target} or {logs}')

    def rewrite(text):
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    for rd in rounds:
        target = destination / rd.name
        logs = sg / 'output' / 'ml_wl' / rd.name
        data = work / 'ml_wl' / rd.name
        target.parent.mkdir(parents=True, exist_ok=True)
        logs.parent.mkdir(parents=True, exist_ok=True)
        if data.exists():
            shutil.move(str(data), str(target))
        else:
            target.mkdir()
        for item in list(rd.iterdir()):
            dst = logs if item.name == 'logs' else target / item.name
            if item.is_symlink() and dst.exists() and item.resolve() == dst.resolve():
                item.unlink()
            elif item.is_symlink() and str(item.readlink()).startswith(str(data) + '/'):
                if not dst.exists():
                    raise FileNotFoundError(dst)
                item.unlink()
            else:
                if dst.exists():
                    raise FileExistsError(dst)
                shutil.move(str(item), str(dst))
        rd.rmdir()
        for p in target.rglob('*'):
            if p.is_symlink():
                old = str(p.readlink())
                new = rewrite(old)
                if new != old:
                    p.unlink()
                    p.symlink_to(new)
            elif p.suffix in {'.json', '.sbatch', '.yaml', '.yml', '.sh'}:
                text = p.read_text()
                new = rewrite(text)
                if new != text:
                    p.write_text(new)
        config = target / 'config.resolved.json'
        if config.exists():
            cfg = json.loads(config.read_text())
            cfg.update(rounds=str(destination), round_dir=str(target),
                       data_root=str(target), log_dir=str(logs))
            config.write_text(json.dumps(cfg, indent=2) + '\n')
        for db in target.rglob('*.db'):
            # Optuna checks the saved absolute dataset paths when resuming.
            with sqlite3.connect(db) as conn:
                tables = {row[0] for row in conn.execute('SELECT name FROM sqlite_master WHERE type="table"')}
                if 'study_user_attributes' not in tables:
                    continue
                backup = db.with_suffix('.db.before_relocation')
                with sqlite3.connect(backup) as out:
                    conn.backup(out)
                for rowid, value in conn.execute('SELECT rowid, value_json FROM study_user_attributes').fetchall():
                    new = rewrite(value)
                    if new != value:
                        conn.execute('UPDATE study_user_attributes SET value_json=? WHERE rowid=?', (new, rowid))
        (target / 'relocation.json').write_text(json.dumps(dict(
            previous_round_dir=str(rd), round_dir=str(target), log_dir=str(logs)), indent=2) + '\n')
        print(f'{rd.name}: results={target} logs={logs}', flush=True)
    legacy.rmdir()


if __name__ == '__main__':
    relocate()
