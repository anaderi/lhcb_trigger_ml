from __future__ import division
from __future__ import print_function
from collections import OrderedDict
import copy

import numpy
import pandas
import math
from numpy.random import normal as randn
from sklearn.utils.validation import check_random_state

__author__ = 'Alex Rogozhnikov'



#region Special functions


def to_nan(data):
    assert isinstance(data, pandas.DataFrame)
    return data.replace(-999, numpy.nan)


def from_nan(data):
    assert isinstance(data, pandas.DataFrame)
    return data.replace(numpy.nan, -999)


def to_projections_(phi, eta, pt):
    return pt * numpy.cos(phi), pt * numpy.sin(phi), pt * numpy.sinh(eta)


def to_projections(df, prefix):
    df[prefix+'px'], df[prefix+'py'], df[prefix+'pz'] = \
        to_projections_(df[prefix+'phi'], df[prefix+'eta'], df[prefix+'pt'])


def from_projections_(px, py, pz):
    p_t = numpy.sqrt(px*px + py*py)
    return numpy.arctan2(py, px), numpy.arcsinh(pz / p_t), p_t


def from_projections(df, prefix):
    df[prefix+'phi'], df[prefix+'eta'], df[prefix+'pt'] = \
        from_projections_(df[prefix+'px'], df[prefix+'py'], df[prefix+'pz'])


def test_projections(size=100):
    triple = (randn(size=size), randn(size=size), randn(size=size))
    res = to_projections_(*from_projections_(*triple))

    for i in [0, 1, 2]:
        assert numpy.allclose(res[i], triple[i])

test_projections()


def compute_energy(m, px, py, pz):
    return numpy.sqrt(m*m + px*px + py*py + pz*pz)


def compute_mass(E, px, py, pz):
    return numpy.sqrt(E*E - px*px - py*py - pz*pz)


def dot(df, prefix1, prefix2):
    return df[prefix1 + "px"] * df[prefix2 + "px"] \
         + df[prefix1 + "py"] * df[prefix2 + "py"] \
         + df[prefix1 + "pz"] * df[prefix2 + "pz"]


def compute_cos(df, prefix1, prefix2):
    return dot(df, prefix1, prefix2) / numpy.sqrt(dot(df, prefix1, prefix1) * dot(df, prefix2, prefix2))


def compute_proj(df, prefix1, prefix2):
    return dot(df, prefix1, prefix2) / numpy.sqrt(dot(df, prefix1, prefix1))


def compute_volume(df, p1, p2, p3):
    return (
          df[p1 + 'x'] * df[p2 + 'y'] * df[p3 + 'z']
        + df[p1 + 'y'] * df[p2 + 'z'] * df[p3 + 'x']
        + df[p1 + 'z'] * df[p2 + 'x'] * df[p3 + 'y']
        - df[p1 + 'x'] * df[p2 + 'z'] * df[p3 + 'y']
        - df[p1 + 'z'] * df[p2 + 'y'] * df[p3 + 'x']
        - df[p1 + 'y'] * df[p2 + 'x'] * df[p3 + 'z'])


def add_not_nan(df, target, source):
    df[target + 'px'] += numpy.nan_to_num(df[source + 'px'])
    df[target + 'py'] += numpy.nan_to_num(df[source + 'py'])
    df[target + 'pz'] += numpy.nan_to_num(df[source + 'pz'])


def to_normal_angle(data, to_symmetric=True, *columns):
    """ Drives columns to range [-pi, pi] if symmetric
    or to [0, 2pi] if not"""
    for column in columns:
        assert column in data.columns, 'no such column ' + str(column)
        if to_symmetric:
            data[column] = ((data[column] + numpy.pi) % (2 * numpy.pi)) - numpy.pi
        else:
            data[column] %= 2 * numpy.pi


#endregion

def add_new_columns(df, include_proj=False, include_hidden=False):
    df = to_nan(df)

    df['MY_total_px'] = 0.
    df['MY_total_py'] = 0.
    df['MY_total_pz'] = 0.

    prefixes = {'lep': 'PRI_lep_', 'tau': 'PRI_tau_', 'ljet': 'PRI_jet_leading_', 'sljet': 'PRI_jet_subleading_'}

    for pf, prefix in prefixes.iteritems():
        to_projections(df, prefix)
        add_not_nan(df, 'MY_total_', prefix)

    from_projections(df, 'MY_total_')

    df['MY_leptau_px'] = df['PRI_lep_px'] + df['PRI_tau_px']
    df['MY_leptau_py'] = df['PRI_lep_py'] + df['PRI_tau_py']
    df['MY_leptau_pz'] = df['PRI_lep_pz'] + df['PRI_tau_pz']

    from_projections(df, 'MY_leptau_')

    if include_hidden:
        # here masses were ignored
        df['PRI_met_px'] = numpy.cos(df['PRI_met_phi']) * df['PRI_met']
        df['PRI_met_py'] = numpy.sin(df['PRI_met_phi']) * df['PRI_met']

        df['MY_hidden_px'] = df['PRI_met_px'] + df['MY_leptau_px']
        df['MY_hidden_py'] = df['PRI_met_py'] + df['MY_leptau_py']
        df['MY_hidden_pz'] = 0
        from_projections(df, 'MY_hidden_')
        df = df.drop(['MY_hidden_pz', 'MY_hidden_eta'], axis=1)

    for pf, prefix in prefixes.iteritems():
        df['MY_' + pf + '_total_cos'] = compute_cos(df, 'MY_total_', prefix)
        df['MY_' + pf + '_total_proj'] = compute_proj(df, 'MY_total_', prefix)

    for pf, prefix in prefixes.iteritems():
        df['MY_' + pf + '_p'] = df[prefix + 'pt'] * numpy.cosh(df[prefix + 'eta'])

    df['MY_total_p'] = df['MY_total_pt'] * numpy.cosh(df['MY_total_eta'])
    # next line is nearly senseless
    df['MY_total_E'] = numpy.sqrt(df['DER_mass_MMC'] ** 2 + df['MY_total_p'] ** 2)
    df['MY_leptau_p'] = df['MY_leptau_pt'] * numpy.cosh(df['MY_leptau_eta'])
    df['MY_leptau_E'] = numpy.sqrt(df['MY_leptau_p'] ** 2 + df['DER_mass_vis'] ** 2)

    for i1, (pf1, prefix1) in enumerate(prefixes.iteritems()):
        for i2, (pf2, prefix2) in enumerate(prefixes.iteritems()):
            if i1 > i2:
                df['MY_' + pf1 + '_' + pf2 + '_cos'] = compute_cos(df, prefix1, prefix2)
            if include_proj:
                if i1 != i2:
                    df['MY_' + pf1 + '_' + pf2 + '_proj'] = compute_proj(df, prefix1, prefix2)

    df = from_nan(df)
    return df


def sum_projections(df, result_p, *prefixes):
    for axis in ['px', 'py', 'pz']:
        df[result_p + axis] = 0.
        for prefix in prefixes:
            df[result_p + axis] += df[prefix + axis]
    return df


def add_columns2(df, include_hidden=True, include_proj=True, include_phi_cos=True):
    m_tau = 1.777
    df = to_nan(df)

    prefixes = OrderedDict()
    prefixes['lep'] = 'PRI_lep_'
    prefixes['tau'] = 'PRI_tau_'
    prefixes['ljet'] = 'PRI_jet_leading_'
    ext_prefixes = copy.copy(prefixes)
    prefixes['sljet'] = 'PRI_jet_subleading_'

    for pf, prefix in prefixes.iteritems():
        to_projections(df, prefix)

    sum_projections(df, 'MY_leptau_', 'PRI_lep_', 'PRI_tau_')
    from_projections(df, 'MY_leptau_')

    if include_hidden:
        # here masses were ignored
        df['PRI_met_px'] = numpy.cos(df['PRI_met_phi']) * df['PRI_met']
        df['PRI_met_py'] = numpy.sin(df['PRI_met_phi']) * df['PRI_met']
        df['PRI_met_pz'] = 0

        df['MY_hidden_px'] = df['PRI_met_px'] + df['MY_leptau_px']
        df['MY_hidden_py'] = df['PRI_met_py'] + df['MY_leptau_py']
        df['MY_hidden_pz'] = 0
        from_projections(df, 'MY_hidden_')
        df = df.drop(['MY_hidden_pz', 'MY_hidden_eta'], axis=1)

    for pf, prefix in prefixes.iteritems():
        df['MY_' + pf + '_total_cos'] = compute_cos(df, 'MY_leptau_', prefix)
        df['MY_' + pf + '_total_proj'] = compute_proj(df, 'MY_leptau_', prefix)
        df['MY_' + pf + '_abs_eta'] = numpy.abs(df[prefix + 'eta'])

    for pf, prefix in prefixes.iteritems():
        df['MY_' + pf + '_P'] = df[prefix + 'pt'] * numpy.cosh(df[prefix + 'eta'])

    df['MY_leptau_P'] = df['MY_leptau_pt'] * numpy.cosh(df['MY_leptau_eta'])
    df['MY_leptau_E'] = numpy.sqrt(df['MY_leptau_P'] ** 2 + df['DER_mass_vis'] ** 2)
    # approximate, neutrinos lost
    df['MY_total_E'] = numpy.sqrt(df['DER_mass_MMC'] ** 2 + df['MY_leptau_P'] ** 2)
    df['MY_hidden2_E'] = df['MY_total_E'] - df['MY_leptau_E']

    for i1, (pf1, prefix1) in enumerate(prefixes.iteritems()):
        for i2, (pf2, prefix2) in enumerate(prefixes.iteritems()):
            if i1 > i2:
                df['MY_' + pf1 + 'and' + pf2 + '_cos'] = compute_cos(df, prefix1, prefix2)
            if i1 != i2 and include_proj:
                df['MY_' + pf1 + 'and' + pf2 + '_proj'] = compute_proj(df, prefix1, prefix2)

    for i1, (pf1, prefix1) in enumerate(ext_prefixes.iteritems()):
        for i2, (pf2, prefix2) in enumerate(ext_prefixes.iteritems()):
            if i1 > i2 and include_phi_cos:
                df['MY_' + pf1 + 'and' + pf2 + '_diffphi'] = numpy.cos(df[prefix1 + 'phi'] - df[prefix2 + 'phi'])

    df['MY_tau_E'] = numpy.sqrt(df['MY_tau_P'] ** 2 + m_tau ** 2)
    df['MY_lep_E'] = df['MY_leptau_E'] - df['MY_tau_E']
    df['MY_lep_msqlike'] = df['MY_lep_E'] ** 2 - df['MY_lep_P'] ** 2

    for axis in ['px', 'py', 'pz']:
        df['MY_leptau_diff_' + axis] = df['PRI_lep_' + axis] - df['PRI_tau_' + axis]
    from_projections(df, 'MY_leptau_diff_')
    df['MY_leptau_diff_phi'] %= 2 * numpy.pi
    df['MY_leptau_diff_P'] = df['MY_leptau_diff_pt'] * numpy.cosh(df['MY_leptau_diff_eta'])

    # summing jets
    sum_projections(df, 'MY_jets_', 'PRI_jet_leading_', 'PRI_jet_subleading_')
    from_projections(df, 'MY_jets_')
    df['MY_jets_P'] = df['MY_jets_pt'] * numpy.cosh(df['MY_jets_eta'])
    df['MY_jets_E'] = numpy.sqrt(df['DER_mass_jet_jet'] ** 2 + df['MY_jets_P'] ** 2)
    df['MY_leptau_E2'] = df['MY_lep_P'] + df['MY_tau_P']

    # data['MY_mmc_modulus'] = - numpy.abs(data['DER_mass_MMC'] - 120)
    df['MY_strange_psum'] = df.MY_leptau_px + df.PRI_tau_pt
    df['MY_leptau_cosphi'] = numpy.cos(df['MY_leptau_phi'])
    df['MY_strange1'] = df['PRI_tau_phi'] ** 2 - df['MY_leptau_phi'] ** 2
    df['MY_strange2'] = numpy.cos(numpy.log(df.MY_leptau_P)) - df['MY_tauandlep_cos']
    df['MY_strange3'] = numpy.log(df.MY_leptau_E) - 0.7 * numpy.exp(df.MY_tauandlep_cos)
    df['MY_strange4'] = df.MY_tau_total_cos - df.MY_tauandlep_cos
    df['MY_strange5'] = df.MY_lepandtau_proj - df.PRI_tau_pz
    df['MY_strange6'] = df.MY_leptau_diff_phi + df.DER_met_phi_centrality
    df['MY_strange7'] = df.MY_ljetandtau_diffphi - df.MY_tauandlep_diffphi
    df['MY_strange_vol1'] = compute_volume(df, 'PRI_met_p', 'PRI_lep_p', 'PRI_tau_p')
    df['MY_strange_vol2'] = compute_volume(df, 'MY_hidden_p', 'PRI_lep_p', 'PRI_tau_p')
    # this is awesome feature
    # data['MY_massdiff'] = (data.DER_mass_transverse_met_lep - 100) ** 2 - (data.MY_leptau_diff_pt - 100) ** 2

    return from_nan(df)


def add_columns3(df, include_hidden=True, include_proj=True):
    df = to_nan(df)

    prefixes = OrderedDict()
    prefixes['lep'] = 'PRI_lep_'
    prefixes['tau'] = 'PRI_tau_'
    prefixes['ljet'] = 'PRI_jet_leading_'
    ext_prefixes = copy.copy(prefixes)
    prefixes['sljet'] = 'PRI_jet_subleading_'

    # making total
    df['MY_total_px'] = 0.
    df['MY_total_py'] = 0.
    df['MY_total_pz'] = 0.
    for pf, prefix in prefixes.iteritems():
        to_projections(df, prefix)
        add_not_nan(df, 'MY_total_', prefix)
    from_projections(df, 'MY_total_')

    sum_projections(df, 'MY_leptau_', 'PRI_lep_', 'PRI_tau_')
    from_projections(df, 'MY_leptau_')

    if include_hidden:
        df['PRI_met_px'] = numpy.cos(df['PRI_met_phi']) * df['PRI_met']
        df['PRI_met_py'] = numpy.sin(df['PRI_met_phi']) * df['PRI_met']

        df['MY_hidden_px'] = df['PRI_met_px'] + df['MY_leptau_px']
        df['MY_hidden_py'] = df['PRI_met_py'] + df['MY_leptau_py']
        df['MY_hidden_pz'] = 0
        from_projections(df, 'MY_hidden_')
        df = df.drop(['MY_hidden_pz', 'MY_hidden_eta'], axis=1)

    for pf, prefix in prefixes.iteritems():
        df['MY_' + pf + '_total_cos'] = compute_cos(df, 'MY_total_', prefix)
        df['MY_' + pf + '_total_proj'] = compute_proj(df, 'MY_total_', prefix)
        df['MY_' + pf + '_p'] = df[prefix + 'pt'] * numpy.cosh(df[prefix + 'eta'])


    df['MY_total_p'] = df['MY_total_pt'] * numpy.cosh(df['MY_total_eta'])
    # next line is nearly senseless
    df['MY_total_E'] = numpy.sqrt(df['DER_mass_MMC'] ** 2 + df['MY_total_p'] ** 2)
    df['MY_leptau_p'] = df['MY_leptau_pt'] * numpy.cosh(df['MY_leptau_eta'])
    df['MY_leptau_E'] = numpy.sqrt(df['MY_leptau_p'] ** 2 + df['DER_mass_vis'] ** 2)

    for i1, (pf1, prefix1) in enumerate(ext_prefixes.iteritems()):
        for i2, (pf2, prefix2) in enumerate(ext_prefixes.iteritems()):
            if i1 > i2:
                df['MY_' + pf1 + '_' + pf2 + '_cos'] = compute_cos(df, prefix1, prefix2)
            if include_proj:
                if i1 != i2:
                    df['MY_' + pf1 + '_' + pf2 + '_proj'] = compute_proj(df, prefix1, prefix2)

    # difference lep - tau
    for axis in ['px', 'py', 'pz']:
        df['MY_leptau_diff_' + axis] = df['PRI_lep_' + axis] - df['PRI_tau_' + axis]
    from_projections(df, 'MY_leptau_diff_')
    df['MY_leptau_diff_phi'] %= 2 * numpy.pi
    df['MY_leptau_diff_P'] = df['MY_leptau_diff_pt'] * numpy.cosh(df['MY_leptau_diff_eta'])

    # jets
    sum_projections(df, 'MY_jets_', 'PRI_jet_leading_', 'PRI_jet_subleading_')
    from_projections(df, 'MY_jets_')
    df['MY_jets_P'] = df['MY_jets_pt'] * numpy.cosh(df['MY_jets_eta'])
    df['MY_jets_E'] = numpy.sqrt(df['DER_mass_jet_jet'] ** 2 + df['MY_jets_P'] ** 2)
    df['MY_sum_E'] = df['MY_jets_E'] + df['MY_leptau_p']
    df['MY_total_leptau_cos'] = compute_cos(df, 'MY_leptau_', 'MY_total_')

    df = from_nan(df)
    return df



def rotate_and_mirror(X, use_mirror_phi=True, use_mirror_eta=False, to_symmetrized_angle=False, base_phi='PRI_met_phi'):
    data = pandas.DataFrame.copy(to_nan(X))
    starting_phi = numpy.copy(numpy.array(data[base_phi]))
    phi_columns = [col for col in data.columns if col.endswith('_phi')]
    eta_columns = [col for col in data.columns if col.endswith('_eta')]

    for column in phi_columns:
        data[column] -= starting_phi
        data[column] %= 2 * math.pi

    if use_mirror_phi:
        mirror_mask = data.PRI_tau_phi > math.pi
        data.loc[mirror_mask, phi_columns] *= -1
        to_normal_angle(data, to_symmetrized_angle, *phi_columns)

    if use_mirror_eta:
        mirror_mask = data.PRI_lep_eta <= 0.
        data.loc[mirror_mask, eta_columns] *= -1

    data = from_nan(data)
    return data


def select_columns(df):
    selected_columns = [col for col in df.columns if not(col.endswith('px') or col.endswith('py') or col.endswith('pz'))]
    return df.loc[:, selected_columns]


def enhance_data(raw_data):
    result = rotate_and_mirror(add_new_columns(raw_data))
    return select_columns(result)


def extend_data(raw_data):
    data = rotate_and_mirror(raw_data, to_symmetrized_angle=True, use_mirror_eta=True)
    data = add_new_columns(data, include_proj=True, include_hidden=True)
    return data


def extend_data2(raw_data):
    data = rotate_and_mirror(raw_data, to_symmetrized_angle=True, use_mirror_eta=True)
    data = add_columns2(data, include_proj=True, include_hidden=True)
    return data


def extend_data3(raw_data):
    data = rotate_and_mirror(raw_data, to_symmetrized_angle=True, use_mirror_eta=True)
    data = add_columns3(data, include_proj=True, include_hidden=True)
    return data

#region smearing + fake multiply


def shuffled_indices(n_samples, shuffle_factor, random_state=None):
    random_state = check_random_state(random_state)
    order = numpy.arange(n_samples) + random_state.normal(0, shuffle_factor * n_samples, size=n_samples)
    quantiles = numpy.argsort(order) + random_state.normal(0, 1, size=n_samples)
    return numpy.clip(quantiles / (n_samples - 1.), 0, 1)


class Shuffler:
    def __init__(self, X, random_state=None):
        X = pandas.DataFrame(X)
        X = to_nan(X)
        self.is_finite = {}
        self.sorted_finite = {}
        self.finite_positions = {}
        self.random_state = check_random_state(random_state)
        for column in X.columns:
            values = X[column]
            is_finite = numpy.isfinite(values)
            self.is_finite[column] = is_finite
            finite = numpy.array(values[is_finite])
            self.sorted_finite[column] = numpy.sort(finite)
            finite_positions = numpy.argsort(numpy.argsort(finite))
            self.finite_positions[column] = finite_positions
        self.X = from_nan(X)

    def generate(self, shuffle_factor=0.1, oldX=None):
        if shuffle_factor <= 0:
            # NB: not copying it, just passing
            return self.X

        if oldX is None:
            result = self.X.copy()
        else:
            result = oldX
        for column in result.columns:
            if column == 'PRI_jet_num':
                continue

            positions = numpy.array(self.finite_positions[column], dtype=float)

            new_position = positions
            new_position += self.random_state.normal(0, len(positions) * shuffle_factor, size=len(positions))

            upper = len(new_position) - 1

            new_position = numpy.abs(new_position)
            new_position = upper - numpy.abs(new_position - upper)
            new_position = numpy.clip(new_position.astype(numpy.int), 0, upper)
            new_finite = self.sorted_finite[column][new_position]

            # new_position /= upper
            # new_position = numpy.clip(new_position, 0., 1.)
            # new_finite = commonutils.weighted_percentile(sorted_finite, new_position, array_sorted=True)
            result.ix[self.is_finite[column], column] = new_finite
        return result


def indices_of_values(array):
    indices = numpy.argsort(array)
    sorted_array = array[indices]
    diff = numpy.nonzero(numpy.ediff1d(sorted_array))[0]
    limits = [0] + list(diff + 1) + [len(array)]
    for i in range(len(limits) - 1):
        yield sorted_array[limits[i]], indices[limits[i]: limits[i+1]]


def check_indices():
    for max_val in [1, 5, 20, 10000]:
        a = numpy.random.randint(0, max_val, 1000)
        for val, indices in indices_of_values(a):
            assert numpy.all(numpy.sort(indices) == numpy.where(a==val)[0])

check_indices()



def fake_multiply(X, y, w, times=2, smearing=0.01, random_state=None):
    """Copies the data n times, one is left for a long"""
    X_parts = []
    y_parts = []
    w_parts = []
    shuffler = Shuffler(X, random_state=random_state)
    for i in range(times):
        X_parts.append(shuffler.generate(smearing))
        y_parts.append(y)
        w_parts.append(w)

    return pandas.concat(X_parts, ignore_index=True), numpy.concatenate(y_parts), numpy.concatenate(w_parts)


#endregion



