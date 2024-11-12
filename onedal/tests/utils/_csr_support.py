# ===============================================================================
# Copyright 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import pytest
import numpy as np
import scipy.sparse as sp

from sklearnex import get_config

from ...utils._dpep_helpers import dpctl_available, dpnp_available

if dpctl_available:
    import dpctl.tensor as dpt

if dpnp_available:
    import dpnp

from onedal.tests.utils._device_selection import get_queues

def get_csr_data_and_queues(
    csr_filter_="csr_array,dpnp,dpctl", device_filter_="cpu,gpu"
):
    """Get supported data in CSR layout for testing.

    This is meant to be used for testing purposes only.

    Parameters
    ----------
    csr_filter_ : str, default="csr_array,dpnp,dpctl"
        Configure output pytest.params for the certain CSR data formats.
    device_filter_ : str, default="cpu,gpu"
        Configure output pytest.params with certain sycl queue for the CSR data,
        where it is applicable.

    Returns
    -------
    list[pytest.param]
        The list of pytest params, included CSR data structure name (str),
        sycl queue, if applicable for the test case, and test
        case id (str).

    Notes
    -----
        Do not use filters for the test cases disabling. Use `pytest.skip`
        or `pytest.xfail` instead.

    See Also
    --------
    _convert_to_csr : Convert input object to certain CSR data structure format.
    """
    csr_data_and_queues = []

    def get_csr_and_q(csr_data: str):
        csr_and_q = []
        for queue in get_queues(device_filter_):
            if queue:
                id = "{}-{}".format(csr_data, queue.id)
                csr_and_q.append(pytest.param(csr_data, queue.values[0], id=id))
        return csr_and_q

    if "csr_array" in csr_filter_:
        csr_data_and_queues.append(pytest.param("csr_array", None, id="csr_array"))
    if dpctl_available and "dpctl" in csr_filter_:
        csr_data_and_queues.extend(get_csr_and_q("dpctl"))
    if dpnp_available and "dpnp" in csr_filter_:
        csr_data_and_queues.extend(get_csr_and_q("dpnp"))

    return csr_data_and_queues

def _convert_to_csr_array(values, col_indices, row_offs, shape, sycl_queue=None, target_fmt=None):
    """Converted input object to certain CSR format."""
    if target_fmt is None or target_fmt == "csr_array":
        # default or scipy.sparse.csr_array.
        # `sycl_queue` arg is ignored.
        return sp.csr_array((values, col_indices, row_offs), shape=shape)
    # scipy.sparse.csr_array on top of DPNP ndarray.
    elif target_fmt == "dpnp":
        values_dpnp = dpnp.asarray(
            values, usm_type="device", sycl_queue=sycl_queue
        )
        col_indices_dpnp = dpnp.asarray(
            col_indices, usm_type="device", sycl_queue=sycl_queue
        )
        row_offs_dpnp = dpnp.asarray(
            row_offs, usm_type="device", sycl_queue=sycl_queue
        )
        return sp.csr_array((values_dpnp, col_indices_dpnp, row_offs_dpnp), shape=shape)
    elif target_fmt == "dpctl":
        # DPCtl tensor.
        values_dpt = dpt.asarray(
            values, usm_type="device", sycl_queue=sycl_queue
        )
        col_indices_dpt = dpt.asarray(
            col_indices, usm_type="device", sycl_queue=sycl_queue
        )
        row_offs_dpt = dpt.asarray(
            row_offs, usm_type="device", sycl_queue=sycl_queue
        )
        return sp.csr_array((values_dpt, col_indices_dpt, row_offs_dpt), shape=shape)

    raise RuntimeError("Unsupported CSR data type conversion")
