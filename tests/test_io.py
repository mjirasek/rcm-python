from __future__ import annotations

import numpy as np

from rcm_python.io import read_atom_groups, read_connectivity, read_csv_matrix, read_xyz


def test_read_xyz_handles_standard_header_and_atomic_numbers(tmp_path) -> None:
    xyz_file = tmp_path / "mini.xyz"
    xyz_file.write_text(
        "3\n"
        "comment\n"
        "6 0.0 0.0 0.0\n"
        "1 1.0 0.0 0.0\n"
        "Zn 0.0 1.0 0.0\n"
    )

    geom = read_xyz(xyz_file)

    assert geom.symbols == ["C", "H", "Zn"]
    assert np.allclose(geom.xyz, [[0, 0, 0], [1, 0, 0], [0, 1, 0]])


def test_read_csv_matrix_and_atom_groups_treat_blanks_as_padding(tmp_path) -> None:
    group_file = tmp_path / "groups.csv"
    group_file.write_text("1,2,\n3,,4\n")

    matrix = read_csv_matrix(group_file)
    groups = read_atom_groups(group_file)

    assert np.allclose(matrix, [[1, 2, 0], [3, 0, 4]])
    assert [group.tolist() for group in groups] == [[0, 2], [1], [3]]


def test_read_connectivity_reverse_keeps_historical_swap_plus_row_reverse(tmp_path) -> None:
    conn_file = tmp_path / "conn.csv"
    conn_file.write_text("1,1,2\n0.5,2,3\n")

    conn = read_connectivity(conn_file)
    reversed_conn = read_connectivity(conn_file, reverse=True)

    assert np.allclose(conn, [[1.0, 1.0, 2.0], [0.5, 2.0, 3.0]])
    assert np.allclose(reversed_conn, [[0.5, 3.0, 2.0], [1.0, 2.0, 1.0]])
