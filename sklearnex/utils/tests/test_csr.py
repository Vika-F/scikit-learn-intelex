# ==============================================================================
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
# ==============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._csr_support import (
    _convert_to_csr_array,
    get_csr_data_and_queues,
)

@pytest.mark.skipif(
    not sklearn_check_version("1.4"),
    reason="Array API dispatch requires sklearn 1.4 version",
)
@pytest.mark.parametrize(
    "csr_data,queue",
    get_csr_data_and_queues(
        csr_filter_="csr_array,dpctl", device_filter_="cpu,gpu"
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.float32, id=np.dtype(np.float32).name),
        pytest.param(np.float64, id=np.dtype(np.float64).name),
    ],
)
def test_validate_data(csr_data, queue, dtype):
    """Test validate_data with CSR data on CPU and GPU.
    """
    pytest.importorskip("array_api_compat")

    from sklearn import config_context
    from sklearn.base import BaseEstimator

    if sklearn_check_version("1.6"):
        from sklearn.utils.validation import validate_data
    else:
        validate_data = BaseEstimator._validate_data

    from sklearn.utils._array_api import _convert_to_numpy, get_namespace

    values_np = np.asarray([1, 2, 3, 4, 1, 11, 8], dtype=dtype)
    col_indices_np = np.asarray([0, 1, 3, 2, 1, 3, 1], dtype=np.int64)
    row_offsets_np = np.asarray([0, 3, 4, 6, 7], dtype=np.int64)

    X_csr = _convert_to_csr_array(values_np, col_indices_np, row_offsets_np, (4, 4), sycl_queue=queue, target_fmt=csr_data)
    with config_context(array_api_dispatch=True):
        est = BaseEstimator()
        xp, _ = get_namespace(X_csr.data)
        X_csr_res = validate_data(
            est, X_csr, accept_sparse="csr", dtype=[xp.float64, xp.float32]
        )
        assert type(X_csr) == type(X_csr_res)
        assert type(X_csr.data) == type(X_csr_res.data)

        assert_allclose(X_csr.data, X_csr_res.data)
