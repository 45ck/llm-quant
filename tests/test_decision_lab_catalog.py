"""Publication regressions: mismatched evidence cannot silently enter the README."""

import hashlib
import json

import pytest

from llm_quant.decision_lab.catalog import (
    END, START, read_register, render_register, repository_file, update_readme,
)


def register(tmp_path):
    evidence = tmp_path/'results.json'
    evidence.write_text('{"sample": true}\n')
    value = {'schema_version': 1, 'results': [{
        'id': 'reference', 'status': 'synthetic_validation', 'trading_approved': False,
        'summary': 'Fictitious observations, not market evidence',
        'artifacts': [{'path': 'results.json',
                       'sha256': hashlib.sha256(evidence.read_bytes()).hexdigest()}],
    }]}
    path = tmp_path/'register.json'
    path.write_text(json.dumps(value))
    return path, value


def test_generated_table_has_status_and_source(tmp_path):
    path, _ = register(tmp_path)
    table = render_register(read_register(tmp_path, path))
    assert '[reference](results.json)' in table
    assert 'not market evidence' in table
    original = f'before\n{START}\nstale\n{END}\nafter\n'
    updated = update_readme(original, table)
    assert updated.startswith('before\n') and updated.endswith('\nafter\n')
    assert update_readme(updated, table) == updated and 'stale' not in updated


def test_evidence_mutation_fails(tmp_path):
    path, _ = register(tmp_path)
    (tmp_path/'results.json').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        read_register(tmp_path, path)


@pytest.mark.parametrize('change', [
    {'id': '../unsafe'}, {'status': 'validated_live_alpha'}, {'trading_approved': True},
    {'summary': ''}, {'artifacts': []}, {'status': 'forward_observed'},
])
def test_invalid_status_or_evidence_fail(tmp_path, change):
    path, value = register(tmp_path)
    value['results'][0].update(change)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_register(tmp_path, path)


def test_duplicate_id_fails(tmp_path):
    path, value = register(tmp_path)
    value['results'].append(value['results'][0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Duplicate'):
        read_register(tmp_path, path)


@pytest.mark.parametrize('text', ['', START, END, START+START+END, END+'x'+START])
def test_missing_repeated_and_reversed_markers_fail(text):
    with pytest.raises(ValueError):
        update_readme(text, START+'new'+END)


def test_path_escape_and_absolute_path_refused(tmp_path):
    outside = tmp_path.parent/'not-public.json'
    outside.write_text('private')
    with pytest.raises(ValueError):
        repository_file(tmp_path, '../not-public.json')
    with pytest.raises(ValueError):
        repository_file(tmp_path, str(outside))


def test_symlink_escape_refused(tmp_path):
    outside = tmp_path.parent/'not-public-symlink.json'
    outside.write_text('private')
    (tmp_path/'link.json').symlink_to(outside)
    with pytest.raises(ValueError):
        repository_file(tmp_path, 'link.json')


def test_private_local_directory_not_publishable(tmp_path):
    path, value = register(tmp_path)
    (tmp_path/'research-local').mkdir()
    raw = tmp_path/'results.json'
    raw.rename(tmp_path/'research-local/inputs.json')
    value['results'][0]['artifacts'][0]['path'] = 'research-local/inputs.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Private'):
        read_register(tmp_path, path)


@pytest.mark.parametrize("relative", ["./research-local/inputs.json",
                                     "nested/../research-local/inputs.json",
                                     "nested//results.json", "../results.json",
                                     "research-local\\inputs.json"])
def test_noncanonical_paths_cannot_bypass_publication_boundary(tmp_path, relative):
    with pytest.raises(ValueError):
        repository_file(tmp_path, relative)


def test_in_root_symlink_to_private_data_is_refused(tmp_path):
    private = tmp_path / "research-local"
    private.mkdir()
    (private / "inputs.json").write_text("private data")
    (tmp_path / "public.json").symlink_to(private / "inputs.json")
    with pytest.raises(ValueError, match="Symlink"):
        repository_file(tmp_path, "public.json")


def test_environment_file_cannot_be_public_evidence(tmp_path):
    (tmp_path / ".env").write_text("private data")
    with pytest.raises(ValueError, match="Private"):
        repository_file(tmp_path, ".env")
