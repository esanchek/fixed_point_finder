# pylint: disable=missing-function-docstring

"""
to avoid pylint warning: Pylint: Redefining name '...' from outer scope pytest.fixture(name=...)
"""
import copy
import pickle
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from FixedPoints import FixedPoints

#############################################################################################
## Three types of attrs:
##   a) data_attrs: contains specific data of each stored fixed point
##   b) nonspecific_attrs: contains generic info for all fixed points
##   c) dim attrs: info about model hidden_dim, n_inpus, number of fp stored
#############################################################################################

@pytest.fixture(name='nonspec_attrs_default_vals')
def fixture_nonspec_attrs_default_vals():
    return {'dtype': np.float32, 'dtype_complex': np.complex64,
            'tol_unique': 1e-3, 'verbose': False, 'do_alloc_nan': False}

@pytest.fixture(name='nonspec_attrs_non_default_vals')
def fixture_nonspec_attrs_non_default_vals():
    return {'dtype': np.complex64, 'dtype_complex': np.float32,
            'tol_unique': 1, 'verbose': True, 'do_alloc_nan': False}

@pytest.fixture(name='dim_attrs_a_value')
def fixture_dim_attrs_a_value():
    return {'n': 2, 'n_states': 5, 'n_inputs': 4}

@pytest.fixture(name='data_attrs_2fps')
def fixture_data_attrs_2fps():
    return dict(xstar=np.random.rand(10).reshape((2, 5)), x_init=np.random.rand(10).reshape((2, 5)),
                inputs=np.random.rand(8).reshape((2, 4)),
                F_xstar=np.random.rand(10).reshape((2, 5)), qstar=np.random.rand(2, ),
                dq=np.random.rand(2, ), n_iters=np.random.rand(2, ),
                J_xstar=np.random.rand(50).reshape(2, 5, 5),
                eigval_J_xstar=np.random.rand(10).reshape((2, 5)),
                eigvec_J_xstar=np.random.rand(50).reshape((2, 5, 5)), is_stable=np.random.rand(2, ),
                cond_id=np.random.rand(2, ))

@pytest.fixture(name='data_attrs_1fp')
def fixture_data_attrs_1fp():
    return dict(xstar=np.random.rand(5).reshape((1, 5)), x_init=np.random.rand(5).reshape((1, 5)),
                inputs=np.random.rand(4).reshape((1, 4)), F_xstar=np.random.rand(5).reshape((1, 5)),
                qstar=np.random.rand(1, ), dq=np.random.rand(1, ), n_iters=np.random.rand(1, ),
                J_xstar=np.random.rand(25).reshape(1, 5, 5),
                eigval_J_xstar=np.random.rand(5).reshape((1, 5)),
                eigvec_J_xstar=np.random.rand(25).reshape((1, 5, 5)), is_stable=np.random.rand(1, ),
                cond_id=np.random.rand(1, ))

@pytest.fixture(name='path_fps_stored')
def fixture_path_fps_stored(tmp_path):
    filepath = tmp_path / "test.fps"
    fps = FixedPoints(verbose=True)
    content = pickle.dumps(fps.__dict__)
    with open(filepath, 'wb') as file:
        file.write(content)
    return filepath

@pytest.fixture(name='path_fps_stored_without_one_attr')
def fixture_path_fps_stored_without_one_attr(tmp_path, dim_attrs_a_value,
                                             nonspec_attrs_non_default_vals):
    def _path_fps_stored_without_one_attr(deleted_attr):
        filepath = tmp_path / "test.fps"
        nonspec_attrs_non_default_vals.pop('do_alloc_nan')
        fps = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value,
                                             **nonspec_attrs_non_default_vals)
        delattr(fps, deleted_attr)
        content = pickle.dumps(fps.__dict__)
        with open(filepath, 'wb') as file:
            file.write(content)
        return filepath

    return _path_fps_stored_without_one_attr

#############################################################################################
## Class stores two lists:
##   a) _data_attrs: contains the list of spacific attrs.
##   b) _nonspecific_attrs: contains the list of nonspecific attrs.
#############################################################################################

def test_fps_contains_attr_with_the_list_of_specific_data_of_fixed_points(data_attrs_1fp):
    fps = FixedPoints()
    assert getattr(fps, '_data_attrs') == list(data_attrs_1fp.keys())

def test_fps_contains_attr_with_the_list_of_attrs_that_apply_to_all_fixed_points(
        nonspec_attrs_default_vals):
    fps = FixedPoints()
    assert getattr(fps, '_nonspecific_attrs') == list(nonspec_attrs_default_vals.keys())

#############################################################################################
## Class constructor has two different "use" modes:
##   a) ONLY DIMS ATTRS ARE PROVIDED (do_alloc_nan: True)
##         dim attrs: must be provided (a.k.a. n, n_states, n_inputs).
##         specific attrs: initialized to nan arrays with proper shape.
##         non specific attrs: initialized to values given in constructor
##   b) SPECIFIC ATTRS ARE PROVIDED (do_alloc_nan: False)
##         dim attrs: if provided, must be ignored.
##                     Instead, they are calculated from provided specific attrs.
##         specific attrs: initialized to nan arrays with proper shape.
##         non specific attrs: initialized to values given in constructor.
#############################################################################################

def test_nonspecific_attrs_passed_in_constructor_are_always_directly_stored(
        nonspec_attrs_non_default_vals):
    fps = FixedPoints(**nonspec_attrs_non_default_vals)
    for attr_name, value in nonspec_attrs_non_default_vals.items():
        assert getattr(fps, attr_name) == value

def test_nonspecific_attrs_handling_not_depend_on_init_mode(dim_attrs_a_value,
                                                            nonspec_attrs_non_default_vals):
    nonspec_attrs_non_default_vals.pop('do_alloc_nan')
    fps = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value, **nonspec_attrs_non_default_vals)
    for attr_name, value in nonspec_attrs_non_default_vals.items():
        assert getattr(fps, attr_name) == value

def test_nonspecific_attrs_default_value_is_not_none(nonspec_attrs_default_vals):
    for attr_name, default_value in nonspec_attrs_default_vals.items():
        assert getattr(FixedPoints(), attr_name) == default_value

def test_1st_mode_all_dim_attrs_must_be_provided(dim_attrs_a_value):
    for attr_name, value in dim_attrs_a_value.items():
        with pytest.raises(ValueError) as excinfo:
            FixedPoints(do_alloc_nan=True, **{attr_name: value})
            assert attr_name + 'must be provided if do_alloc_nan == True.' == str(excinfo.value)

def test_1st_mode_dims_attrs_are_simply_stored(dim_attrs_a_value):
    fps = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value)
    actual = [getattr(fps, attr_name) for attr_name in dim_attrs_a_value]
    assert actual == list(dim_attrs_a_value.values())

def test_1st_mode_specific_attrs_are_initialized_to_arrays_with_proper_shape(dim_attrs_a_value):
    fps = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value)
    specific_attrs_2d1 = ['xstar', 'x_init', 'F_xstar', 'eigval_J_xstar']
    specific_attrs_2d2 = ['inputs']
    specific_attrs_1d   = ['qstar', 'dq', 'n_iters', 'is_stable', 'cond_id']
    specific_attrs_3d   = ['J_xstar', 'eigvec_J_xstar']
    assert all((getattr(fps, attr).shape == (fps.n, fps.n_states) for attr in specific_attrs_2d1))
    assert all((getattr(fps, attr).shape == (fps.n, fps.n_inputs) for attr in specific_attrs_2d2))
    assert all((getattr(fps, attr).shape == (fps.n,) for attr in specific_attrs_1d))
    assert all((getattr(fps, attr).shape == (fps.n, fps.n_states, fps.n_states)
                for attr in specific_attrs_3d))

@pytest.mark.parametrize('attr_name', ['eigval_J_xstar', 'eigvec_J_xstar'])
def test_1st_mode_dtype_of_the_some_specific_attrs_is_forced_at_inizialization(attr_name,
                                                                               dim_attrs_a_value):
    fps = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value)
    assert getattr(fps, attr_name).dtype == getattr(fps, 'dtype_complex')

def test_1st_mode_specific_attrs_are_initialized_with_nans(dim_attrs_a_value, data_attrs_1fp):
    fps = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value)
    for attr_name in data_attrs_1fp:
        actual = getattr(fps, attr_name)
        assert np.isnan(actual).all()

@pytest.mark.parametrize('attr_name', ['xstar', 'x_init', 'F_xstar', 'J_xstar'])
def test_2nd_mode_when_2d1_specific_attr_are_provided_its_dim_attr_are_calculated(attr_name,
                                                                                  data_attrs_2fps):
    a_value = data_attrs_2fps[attr_name]
    fps = FixedPoints(do_alloc_nan=False, **{attr_name: a_value})
    assert fps.n, fps.n_states == a_value.shape

@pytest.mark.parametrize('attr_name', ['xstar', 'x_init', 'F_xstar', 'J_xstar'])
def test_2nd_mode_when_dim_attr_are_calculated_constructor_args_are_ignored(attr_name,
                                                                            dim_attrs_a_value,
                                                                            data_attrs_1fp):
    a_value = data_attrs_1fp[attr_name]
    fps = FixedPoints(do_alloc_nan=False, **{attr_name: a_value}, **dim_attrs_a_value)
    assert fps.n, fps.n_states == a_value.shape

def test_2nd_mode_when_2d1_specific_attr_are_not_provided_its_dim_attr_are_set_to_none():
    fps = FixedPoints(do_alloc_nan=False)
    assert fps.n is None
    assert fps.n_states is None

def test_2nd_mode_when_attr_inputs_is_provided_n_inputs_is_calculated(data_attrs_1fp):
    a_value = data_attrs_1fp['inputs']
    fps = FixedPoints(do_alloc_nan=False, inputs=a_value)
    assert fps.n_inputs == a_value.shape[1]

def test_2nd_mode_when_attr_inputs_is_not_provided_n_inputs_is_set_to_none():
    fps = FixedPoints(do_alloc_nan=False)
    assert fps.n_inputs is None

def test_2nd_mode_when_attr_inputs_is_provided_n_is_calculated_if_no_other_source_to_obtain_n(
        data_attrs_1fp):
    a_value = data_attrs_1fp['inputs']
    fps = FixedPoints(do_alloc_nan=False, inputs=a_value)
    assert fps.n == a_value.shape[0]

def test_2nd_mode_specific_attrs_are_simply_stored(data_attrs_1fp):
    fps = FixedPoints(do_alloc_nan=False, **data_attrs_1fp)
    for attr_name in data_attrs_1fp:
        assert_array_equal(getattr(fps, attr_name), data_attrs_1fp[attr_name])

def test_2nd_mode_specific_attrs_default_value_is_none(data_attrs_2fps):
    fps = FixedPoints(do_alloc_nan=False)
    for attr_name in data_attrs_2fps:
        assert getattr(fps, attr_name) is None

@pytest.mark.skip('possible future implementation via exception')
def test_after_1st_mode_all_data_attrs_must_reflect_same_number_of_fps():
    pass

#############################################################################################
## Class magic method __equ__ is defined
#############################################################################################
def test_fps_supports_equ_operator(data_attrs_1fp,
                                   dim_attrs_a_value,
                                   nonspec_attrs_non_default_vals):
    fps_1 = FixedPoints(**data_attrs_1fp, **dim_attrs_a_value, **nonspec_attrs_non_default_vals)
    fps_2 = FixedPoints(**data_attrs_1fp, **dim_attrs_a_value, **nonspec_attrs_non_default_vals)
    assert fps_1 == fps_2

def test_two_fps_are_equal_if_all_its_attrs_are_equal(data_attrs_1fp,
                                                      nonspec_attrs_non_default_vals):
    fps_1 = FixedPoints(**data_attrs_1fp)
    fps_2 = FixedPoints(**data_attrs_1fp, **nonspec_attrs_non_default_vals)
    assert fps_1 != fps_2

def test_obviously_equality_is_supported_by_both_initialization_modes(dim_attrs_a_value):
    fps_1 = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value)
    fps_2 = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value)
    assert fps_1 == fps_2

def test_none_vs_nans_nparrays_are_different_for_fps_equality(dim_attrs_a_value):
    fps_1 = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value)
    fps_2 = FixedPoints(do_alloc_nan=False, **dim_attrs_a_value)
    assert fps_1 != fps_2

#############################################################################################
## Class magic methods __setitem__ is defined
#############################################################################################

def test_fps_supports_overwrite_partially_a_fixed_point_by_indexing(data_attrs_2fps,
                                                                    data_attrs_1fp):
    index = 1
    fps_1 = FixedPoints(do_alloc_nan=False, **data_attrs_2fps)
    fps_2 = FixedPoints(do_alloc_nan=False, **data_attrs_1fp)
    fps_1[index] = fps_2
    for attr_name in data_attrs_2fps:
        assert_array_equal(getattr(fps_1[index], attr_name), getattr(fps_2, attr_name))

def test_fps_supports_overwrite_index_must_be_init_mode_agnotic(data_attrs_2fps, data_attrs_1fp):
    index = 1
    fps_1 = FixedPoints(do_alloc_nan=False, **data_attrs_2fps)
    n, n_states = data_attrs_1fp['xstar'].shape
    _, n_inputs = data_attrs_1fp['inputs'].shape
    fps_2 = FixedPoints(do_alloc_nan=True, n=n, n_states=n_states, n_inputs=n_inputs,
                        **data_attrs_1fp)
    fps_1[index] = fps_2
    for attr_name in data_attrs_2fps:
        assert_array_equal(getattr(fps_1[index], attr_name), getattr(fps_2, attr_name))

@pytest.mark.skip('possible future implementation via exception')
def test_while_setting_value_fps_must_be_a_fps():
    pass

#############################################################################################
## Class magic methods __getitem__ is defined
#############################################################################################

def test_fps_supports_single_element_indexing(data_attrs_2fps, nonspec_attrs_default_vals):
    index = 1
    fps = FixedPoints(do_alloc_nan=False, **data_attrs_2fps)
    assert getattr(fps[index], 'n') == 1
    for attr_name in data_attrs_2fps:
        assert_array_equal(getattr(fps[index], attr_name), getattr(fps, attr_name)[[index]])
    for attr_name in ['n_states', 'n_inputs']:
        assert getattr(fps[index], attr_name) == getattr(fps, attr_name)
    for attr_name in ['dtype', 'tol_unique']:
        assert getattr(fps[index], attr_name) == getattr(fps, attr_name)
    for attr_name in ['dtype_complex', 'do_alloc_nan']:
        assert getattr(fps[index], attr_name) == nonspec_attrs_default_vals[attr_name]

def test_none_attrs_are_indexed_as_none(data_attrs_2fps):
    index = 1
    fps = FixedPoints(do_alloc_nan=False, **data_attrs_2fps)
    fps.xstar = None
    assert getattr(fps[index], 'xstar') is None

def test_fps_indexing_must_be_init_mode_agnostic(data_attrs_2fps, dim_attrs_a_value,
                                                 nonspec_attrs_non_default_vals,
                                                 nonspec_attrs_default_vals):
    index = 1
    nonspec_attrs_non_default_vals.pop('do_alloc_nan')
    fps = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value, **nonspec_attrs_non_default_vals)

    assert getattr(fps[index], 'n') == 1
    for attr_name in data_attrs_2fps:
        assert_array_equal(getattr(fps[index], attr_name), getattr(fps, attr_name)[[index]])
    for attr_name in ['n_states', 'n_inputs']:
        assert getattr(fps[index], attr_name) == getattr(fps, attr_name)
    for attr_name in ['dtype', 'tol_unique']:
        assert getattr(fps[index], attr_name) == getattr(fps, attr_name)
    for attr_name in ['dtype_complex', 'do_alloc_nan']:
        assert getattr(fps[index], attr_name) == nonspec_attrs_default_vals[attr_name]


#############################################################################################
## Class magic methods __len__ is defined
#############################################################################################

def test_fps_supports_length_attribute(data_attrs_2fps):
    fps = FixedPoints(do_alloc_nan=False, **data_attrs_2fps)
    assert len(fps) == fps.n

#############################################################################################
## method: find. To locate indexes which match a given fixed point
## magic method: __contains__ is defined based on find
#############################################################################################
@pytest.mark.skip('future implementation via rise exception')
def test_given_fp_to_locate_must_be_a_fp():
    pass

def test_given_a_fp_its_index_in_fps_can_be_found(data_attrs_2fps):
    fps = FixedPoints(do_alloc_nan=False, **data_attrs_2fps)
    for index in range(fps.n):
        assert fps.find(fps[index]) == index
        assert fps[index] in fps

@pytest.mark.parametrize('attr_name', ['n_states', 'n_inputs'])
def test_to_find_a_fp_its_dim_attrs_must_be_compatible_with_fps_dim_attrs(attr_name,
                                                                          dim_attrs_a_value,
                                                                          data_attrs_2fps):
    fps = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value, **data_attrs_2fps)
    fp_in_index0 = fps[0]
    actual_value = getattr(fp_in_index0, attr_name)
    setattr(fp_in_index0, attr_name, actual_value*100)

    assert_array_equal(fps.find(fp_in_index0), np.array([], dtype=int))
    assert fp_in_index0 not in fps

def test_fp_match_criteria_is_only_based_on_inputs_and_xstar():
    inputs_1, xstar_1 = np.array([[0, 0]]), np.array([[1, 1]])
    inputs_2, xstar_2 = np.array([[0, 0]]), np.array([[1, 1]])

    fps_1 = FixedPoints(do_alloc_nan=False, xstar=xstar_1, inputs=inputs_1)
    fps_2 = FixedPoints(do_alloc_nan=False, xstar=xstar_2, inputs=inputs_2)

    assert fps_1.find(fps_2) == 0
    assert fps_2 in fps_1

@pytest.mark.parametrize('distance_squared, tolerance, expected', [(1, 2, 0),
                                                                   (2, 1, np.array([], dtype=int))])
def test_fp_match_criteria_is_l2_less_than_tol_unique(distance_squared, tolerance, expected):
    inputs_1, xstar_1 = np.array([[0, 0]]), np.array([[1, 1]])
    inputs_2, xstar_2 = np.array([[0, 0]]), np.array([[1, 1 + distance_squared]])

    fps_1 = FixedPoints(do_alloc_nan=False, xstar=xstar_1, inputs=inputs_1)
    fps_2 = FixedPoints(do_alloc_nan=False, xstar=xstar_2, inputs=inputs_2)
    setattr(fps_1, 'tol_unique', tolerance)

    assert_array_equal(fps_1.find(fps_2), expected)

def test_if_a_fp_appear_multiple_times_all_indexes_are_located_as_a_nparray(data_attrs_1fp):
    _, n_states = data_attrs_1fp['xstar'].shape
    _, n_inputs = data_attrs_1fp['inputs'].shape

    fps = FixedPoints(do_alloc_nan=True, n=3, n_states=n_states, n_inputs=n_inputs)
    fp_in_index0_2 = FixedPoints(**data_attrs_1fp)
    fps[0] = fp_in_index0_2
    fps[1] = FixedPoints()
    fps[2] = fp_in_index0_2

    assert_array_equal(fps.find(fp_in_index0_2), np.array([0, 2], dtype=int))
    assert fp_in_index0_2 in fps

#############################################################################################
# update method, combines entries from another fps into this object
#############################################################################################

def test_fps_allows_combining_entries_from_another_fps(data_attrs_2fps):
    fps = FixedPoints(**data_attrs_2fps)
    previous_n = fps.n
    original_fps_0 = copy.deepcopy(fps[0])
    original_fps_1 = copy.deepcopy(fps[1])
    fps.update(fps)

    assert fps.n == previous_n + 2
    assert (fps[0], fps[1]) == (original_fps_0, original_fps_1)
    assert (fps[2], fps[3]) == (original_fps_0, original_fps_1)

@pytest.mark.skip('possible future implementation via exception')
def test_all_data_attrs_must_be_in_the_info_source_fps():
    pass

@pytest.mark.skip('possible future implementation via exception')
def test_shapes_after_combining_must_match():
    pass

@pytest.mark.skip('possible future implementation via exception')
def test_to_combine_both_non_specific_shapes_must_match():
    pass

#############################################################################################
# I/O operations
#############################################################################################

def test_fps_can_be_stored_and_read_to_disk(tmp_path, data_attrs_2fps):
    fps = FixedPoints(**data_attrs_2fps)
    save_path = tmp_path / "test.fps"
    fps.save(save_path)

    fps_read = FixedPoints()
    fps_read.restore(save_path)
    assert fps_read == fps
    assert len(list(tmp_path.iterdir())) == 1

@pytest.mark.parametrize('verbosity, expected_msg', [(True, 'Saving FixedPoints object.\n'),
                                                     (False, '')])
def test_based_on_nonspecific_attr_verbose_fps_save_can_be_verbose(verbosity, expected_msg,
                                                                   tmp_path, capfd):
    fps = FixedPoints(verbose=verbosity)
    save_path = tmp_path / "test.fps"
    fps.save(save_path)

    outs, _ = capfd.readouterr()
    assert outs == expected_msg

@pytest.mark.parametrize('verbosity, expected_msg', [(True, 'Restoring FixedPoints object.\n'),
                                                     (False, '')])
def test_based_on_nonspecific_attr_verbose_fps_restore_can_be_verbose(verbosity, expected_msg,
                                                                      capfd, path_fps_stored):
    filepath = path_fps_stored
    fps = FixedPoints(verbose=verbosity)
    fps.restore(filepath)
    outs, _ = capfd.readouterr()

    assert outs == expected_msg

def test_restore_must_handle_previous_fps_versions_without_attr_do_alloc_nan(
        path_fps_stored_without_one_attr):
    filepath = path_fps_stored_without_one_attr('do_alloc_nan')
    fps = FixedPoints()
    fps.restore(filepath)
    assert fps.do_alloc_nan is False

@pytest.mark.parametrize("attr_name, shape", [('eigval_J_xstar', (2, 5)),
                                              ('eigvec_J_xstar', (2, 5, 5)),
                                              ('is_stable', (2, )),
                                              ('cond_id', (2, ))])
def test_restore_must_handle_previous_fps_versions_without_eigval_j_xstar_attr(
        attr_name, shape, path_fps_stored_without_one_attr):
    filepath = path_fps_stored_without_one_attr('eigval_J_xstar')
    fps = FixedPoints()
    fps.restore(filepath)
    attr_value = getattr(fps, attr_name)
    assert_array_equal(attr_value, np.full(shape, np.NAN))
    if attr_name in ['eigval_J_xstar', 'eigvec_J_xstar']:
        assert attr_value.dtype == np.complex64

@pytest.mark.skip('possible future implementation via exception')
def test_after_update_non_specific_shapes_must_match():
    pass

#############################################################################################
# Print operations
#############################################################################################

def test_print_fps_summary(capfd, data_attrs_2fps):
    fps = FixedPoints(**data_attrs_2fps)
    fps.print_summary()
    outs, _ = capfd.readouterr()
    assert outs == f"\nThe q function at the fixed points:\n{fps.qstar}\n" \
                   f"\nChange in the q function from the final iteration of each optimization:\n" \
                   f"{fps.dq}\n" \
                   f"\nNumber of iterations completed for each optimization:\n{fps.n_iters}\n" \
                   f"\nThe fixed points:\n{fps.xstar}\n" \
                   f"\nThe fixed points after one state transition:\n{fps.F_xstar}\n" \
                   "(these should be very close to the fixed points)\n" \
                   f"\nThe Jacobians at the fixed points:\n{fps.J_xstar}\n"

def test_print_fps_shapes(dim_attrs_a_value, nonspec_attrs_non_default_vals,
                          data_attrs_2fps, capfd):
    nonspec_attrs_non_default_vals.pop('do_alloc_nan')
    fps = FixedPoints(do_alloc_nan=True, **dim_attrs_a_value, **nonspec_attrs_non_default_vals,
                      **data_attrs_2fps)
    fps.print_shapes()
    out, _ = capfd.readouterr()
    list_str = [f"{attr_name}: {attr_value.shape}" for attr_name, attr_value in data_attrs_2fps.items()]
    assert out == "\n".join(list_str) + "\n"

#############################################################################################
# a) Get unique
# b) Transform
# c) Decompose Jacobians
# d) Concatenate

#############################################################################################

"""
@contextmanager
def mock_file(filepath):
    with open(filepath, 'wb') as f:
        content = pickle.dumps(FixedPoints(verbose=True).__dict__)
        f.write(content)
    yield filepath
    try:
        os.remove(filepath)
    except Exception:
        pass

@pytest.mark.parametrize('verbosity, expected_msg', [(True, 'Restoring FixedPoints object.\n'),
                                                     (False, '')])
def test_based_on_nonspecific_attr_verbose_fps_restore_can_be_verbose(verbosity, expected_msg,
                                                                      tmp_path, capfd):
    restore_path = tmp_path / "test.fps"
    with mock_file(restore_path):
        fps = FixedPoints(verbose=verbosity)
        fps.restore(restore_path)
        out, _ = capfd.readouterr()
        assert out == expected_msg

def test_based_on_nonspecific_attr_verbose_fps_restore_can_be_verbose_2(tmp_path, mocker, capfd):
    mck = mocker.patch('FixedPoints.FixedPoints.restore')
    fps = FixedPoints(verbose=True)
    restore_path = tmp_path / "test.fps"
    fps.restore(restore_path)

    out, _ = capfd.readouterr()
    assert out == "hola"
"""