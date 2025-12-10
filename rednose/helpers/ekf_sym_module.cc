
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "helpers/ekf_sym.h"

#include <iostream>
#include <vector>
#include <string>

using namespace EKFS;

// --- Helper Functions ---

static PyObject* vector_to_numpy(const Eigen::VectorXd& vec) {
    npy_intp dims[1] = { (npy_intp)vec.size() };
    PyObject* arr = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!arr) return NULL;

    double* data = (double*)PyArray_DATA((PyArrayObject*)arr);
    Eigen::Map<Eigen::VectorXd>(data, vec.size()) = vec;

    return arr;
}

static PyObject* matrix_to_numpy(const MatrixXdr& mat) {
    npy_intp dims[2] = { (npy_intp)mat.rows(), (npy_intp)mat.cols() };
    PyObject* arr = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!arr) return NULL;

    double* data = (double*)PyArray_DATA((PyArrayObject*)arr);
    Eigen::Map<MatrixXdr>(data, mat.rows(), mat.cols()) = mat;

    return arr;
}

static PyArrayObject* get_contiguous_double_array(PyObject* obj, int min_depth, int max_depth) {
    return (PyArrayObject*)PyArray_ContiguousFromAny(obj, NPY_DOUBLE, min_depth, max_depth);
}

// --- EKFSym Wrapper ---

typedef struct {
    PyObject_HEAD
    EKFSym* ekf;
} EKFSymObject;

static void EKFSym_dealloc(EKFSymObject* self) {
    if (self->ekf) {
        delete self->ekf;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int EKFSym_init(EKFSymObject* self, PyObject* args, PyObject* kwds) {


    char *gen_dir_str, *name_str;
    PyObject *Q_obj, *x_init_obj, *P_init_obj;
    int dim_main, dim_main_err;
    int N = 0;
    int dim_augment = 0;
    int dim_augment_err = 0;
    PyObject *maha_test_kinds_obj = NULL;
    PyObject *quaternion_idxs_obj = NULL;
    PyObject *global_vars_obj = NULL;
    double max_rewind_age = 1.0;
    PyObject *logger_obj = NULL;

    static char* kwlist[] = {
        (char*)"gen_dir", (char*)"name", (char*)"Q", (char*)"x_initial", (char*)"P_initial",
        (char*)"dim_main", (char*)"dim_main_err", (char*)"N", (char*)"dim_augment", (char*)"dim_augment_err",
        (char*)"maha_test_kinds", (char*)"quaternion_idxs", (char*)"global_vars", (char*)"max_rewind_age", (char*)"logger", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ssOOOii|iiiOOOdO", kwlist,
        &gen_dir_str, &name_str, &Q_obj, &x_init_obj, &P_init_obj,
        &dim_main, &dim_main_err, &N, &dim_augment, &dim_augment_err,
        &maha_test_kinds_obj, &quaternion_idxs_obj, &global_vars_obj, &max_rewind_age, &logger_obj)) {
        return -1;
    }

    ekf_load_and_register(std::string(gen_dir_str), std::string(name_str));

    PyArrayObject *Q_arr = get_contiguous_double_array(Q_obj, 2, 2);
    PyArrayObject *x_init_arr = get_contiguous_double_array(x_init_obj, 1, 1);
    PyArrayObject *P_init_arr = get_contiguous_double_array(P_init_obj, 2, 2);

    if (!Q_arr || !x_init_arr || !P_init_arr) {
        Py_XDECREF(Q_arr); Py_XDECREF(x_init_arr); Py_XDECREF(P_init_arr);
        return -1;
    }

    std::vector<int> maha, quat_idxs;
    std::vector<std::string> globals;

    if (maha_test_kinds_obj) {
        PyObject *iter = PyObject_GetIter(maha_test_kinds_obj);
        if (!iter) goto fail;
        PyObject *item;
        while ((item = PyIter_Next(iter))) {
            maha.push_back((int)PyLong_AsLong(item));
            Py_DECREF(item);
        }
        Py_DECREF(iter);
        if (PyErr_Occurred()) goto fail;
    }

    if (quaternion_idxs_obj) {
        PyObject *iter = PyObject_GetIter(quaternion_idxs_obj);
        if (!iter) goto fail;
        PyObject *item;
        while ((item = PyIter_Next(iter))) {
            quat_idxs.push_back((int)PyLong_AsLong(item));
            Py_DECREF(item);
        }
        Py_DECREF(iter);
         if (PyErr_Occurred()) goto fail;
    }

    if (global_vars_obj) {
        PyObject *iter = PyObject_GetIter(global_vars_obj);
        if (!iter) goto fail;
        PyObject *item;
        while ((item = PyIter_Next(iter))) {
            const char* s = PyUnicode_AsUTF8(item);
            if (s) globals.push_back(std::string(s));
            Py_DECREF(item);
        }
        Py_DECREF(iter);
        if (PyErr_Occurred()) goto fail;
    }

    {
        Eigen::Map<MatrixXdr> Q_map((double*)PyArray_DATA(Q_arr), PyArray_DIM(Q_arr, 0), PyArray_DIM(Q_arr, 1));
        Eigen::Map<Eigen::VectorXd> x_map((double*)PyArray_DATA(x_init_arr), PyArray_DIM(x_init_arr, 0));
        Eigen::Map<MatrixXdr> P_map((double*)PyArray_DATA(P_init_arr), PyArray_DIM(P_init_arr, 0), PyArray_DIM(P_init_arr, 1));

        self->ekf = new EKFSym(
            std::string(name_str), Q_map, x_map, P_map,
            dim_main, dim_main_err, N, dim_augment, dim_augment_err,
            maha, quat_idxs, globals, max_rewind_age
        );
    }

    Py_DECREF(Q_arr);
    Py_DECREF(x_init_arr);
    Py_DECREF(P_init_arr);
    return 0;

fail:
    Py_XDECREF(Q_arr);
    Py_XDECREF(x_init_arr);
    Py_XDECREF(P_init_arr);
    return -1;
}

static PyObject* EKFSym_init_state(EKFSymObject* self, PyObject* args) {
    PyObject *state_obj, *covs_obj;
    double filter_time;
    if (!PyArg_ParseTuple(args, "OOd", &state_obj, &covs_obj, &filter_time)) return NULL;

    PyArrayObject *state_arr = get_contiguous_double_array(state_obj, 1, 1);
    PyArrayObject *covs_arr = get_contiguous_double_array(covs_obj, 2, 2);
    if (!state_arr || !covs_arr) {
        Py_XDECREF(state_arr); Py_XDECREF(covs_arr);
        return NULL;
    }

    Eigen::Map<Eigen::VectorXd> state_map((double*)PyArray_DATA(state_arr), PyArray_DIM(state_arr, 0));
    Eigen::Map<MatrixXdr> covs_map((double*)PyArray_DATA(covs_arr), PyArray_DIM(covs_arr, 0), PyArray_DIM(covs_arr, 1));

    self->ekf->init_state(state_map, covs_map, filter_time);

    Py_DECREF(state_arr);
    Py_DECREF(covs_arr);
    Py_RETURN_NONE;
}

static PyObject* EKFSym_state(EKFSymObject* self, PyObject* args) {
    return vector_to_numpy(self->ekf->state());
}

static PyObject* EKFSym_covs(EKFSymObject* self, PyObject* args) {
    return matrix_to_numpy(self->ekf->covs());
}

static PyObject* EKFSym_set_filter_time(EKFSymObject* self, PyObject* args) {
    double t;
    if (!PyArg_ParseTuple(args, "d", &t)) return NULL;
    self->ekf->set_filter_time(t);
    Py_RETURN_NONE;
}

static PyObject* EKFSym_get_filter_time(EKFSymObject* self, PyObject* args) {
    return PyFloat_FromDouble(self->ekf->get_filter_time());
}

static PyObject* EKFSym_set_global(EKFSymObject* self, PyObject* args) {
    char* name;
    double val;
    if (!PyArg_ParseTuple(args, "sd", &name, &val)) return NULL;
    self->ekf->set_global(std::string(name), val);
    Py_RETURN_NONE;
}

static PyObject* EKFSym_reset_rewind(EKFSymObject* self, PyObject* args) {
    self->ekf->reset_rewind();
    Py_RETURN_NONE;
}

static PyObject* EKFSym_predict(EKFSymObject* self, PyObject* args) {
    double t;
    if (!PyArg_ParseTuple(args, "d", &t)) return NULL;
    self->ekf->predict(t);
    Py_RETURN_NONE;
}

// predict_and_update_batch(double t, int kind, vector[MapVectorXd] z, vector[MapMatrixXdr] R, vector[vector[double]] extra_args, bool augment)
static PyObject* EKFSym_predict_and_update_batch(EKFSymObject* self, PyObject* args, PyObject* kwds) {
    double t;
    int kind;
    PyObject *z_obj, *R_obj;
    PyObject *extra_args_obj = NULL;
    int augment = 0;

    static char* kwlist[] = {
        (char*)"t", (char*)"kind", (char*)"z", (char*)"R", (char*)"extra_args", (char*)"augment", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "diOO|Oi", kwlist, &t, &kind, &z_obj, &R_obj, &extra_args_obj, &augment)) {
        return NULL;
    }

    std::vector<PyArrayObject*> z_arrays;
    std::vector<PyArrayObject*> R_arrays;
    std::vector<Eigen::Map<Eigen::VectorXd>> z_maps;
    std::vector<Eigen::Map<MatrixXdr>> R_maps;
    std::vector<std::vector<double>> extra_args_cpp;

    auto cleanup = [&]() {
        for (auto p : z_arrays) Py_DECREF(p);
        for (auto p : R_arrays) Py_DECREF(p);
    };

    PyObject *z_iter = PyObject_GetIter(z_obj);
    if (z_iter) {
        PyObject *item;
        while ((item = PyIter_Next(z_iter))) {
             PyArrayObject* arr = get_contiguous_double_array(item, 1, 1);
             Py_DECREF(item);
             if (!arr) { Py_DECREF(z_iter); cleanup(); return NULL; }
             z_arrays.push_back(arr);
             z_maps.emplace_back((double*)PyArray_DATA(arr), PyArray_DIM(arr, 0));
        }
        Py_DECREF(z_iter);
        if (PyErr_Occurred()) { cleanup(); return NULL; }
    } else {
         PyErr_SetString(PyExc_TypeError, "z must be iterable");
         cleanup(); return NULL;
    }

    PyObject *R_iter = PyObject_GetIter(R_obj);
    if (R_iter) {
        PyObject *item;
        while ((item = PyIter_Next(R_iter))) {
             PyArrayObject* arr = get_contiguous_double_array(item, 2, 2);
             Py_DECREF(item);
             if (!arr) { Py_DECREF(R_iter); cleanup(); return NULL; }
             R_arrays.push_back(arr);
             R_maps.emplace_back((double*)PyArray_DATA(arr), PyArray_DIM(arr, 0), PyArray_DIM(arr, 1));
        }
        Py_DECREF(R_iter);
        if (PyErr_Occurred()) { cleanup(); return NULL; }
    } else {
         PyErr_SetString(PyExc_TypeError, "R must be iterable");
         cleanup(); return NULL;
    }

    if (extra_args_obj) {
        PyObject *ea_iter = PyObject_GetIter(extra_args_obj);
        if (ea_iter) {
            PyObject *inner;
            while ((inner = PyIter_Next(ea_iter))) {
                std::vector<double> ea;
                PyObject *inner_iter = PyObject_GetIter(inner);
                if (inner_iter) {
                     PyObject *val;
                     while ((val = PyIter_Next(inner_iter))) {
                        ea.push_back(PyFloat_AsDouble(val));
                        Py_DECREF(val);
                        if (PyErr_Occurred()) { Py_DECREF(inner_iter); Py_DECREF(inner); Py_DECREF(ea_iter); cleanup(); return NULL; }
                     }
                     Py_DECREF(inner_iter);
                } else {
                     PyErr_Clear();
                }
                extra_args_cpp.push_back(ea);
                Py_DECREF(inner);
            }
            Py_DECREF(ea_iter);
             if (PyErr_Occurred()) { cleanup(); return NULL; }
        }
    } else {
        extra_args_cpp.push_back({});
    }
    if (extra_args_cpp.empty()) extra_args_cpp.push_back({});

    std::optional<Estimate> res = self->ekf->predict_and_update_batch(t, kind, z_maps, R_maps, extra_args_cpp, (bool)augment);

    cleanup(); // arrays no longer needed after call returns result copy

    if (!res.has_value()) {
        Py_RETURN_NONE;
    }

    Estimate& est = res.value();

    PyObject* res_tuple = PyTuple_New(9);

    PyTuple_SetItem(res_tuple, 0, vector_to_numpy(est.xk1));
    PyTuple_SetItem(res_tuple, 1, vector_to_numpy(est.xk));
    PyTuple_SetItem(res_tuple, 2, matrix_to_numpy(est.Pk1));
    PyTuple_SetItem(res_tuple, 3, matrix_to_numpy(est.Pk));
    PyTuple_SetItem(res_tuple, 4, PyFloat_FromDouble(est.t));
    PyTuple_SetItem(res_tuple, 5, PyLong_FromLong(est.kind));

    PyObject* y_list = PyList_New(est.y.size());
    for(size_t i=0; i<est.y.size(); ++i) {
        PyList_SetItem(y_list, i, vector_to_numpy(est.y[i]));
    }
    PyTuple_SetItem(res_tuple, 6, y_list);

    Py_INCREF(z_obj);
    PyTuple_SetItem(res_tuple, 7, z_obj);

    if (extra_args_obj) {
        Py_INCREF(extra_args_obj);
        PyTuple_SetItem(res_tuple, 8, extra_args_obj);
    } else {
        PyObject* def = PyList_New(1);
        PyList_SetItem(def, 0, PyList_New(0));
        PyTuple_SetItem(res_tuple, 8, def);
    }

    return res_tuple;
}

static PyMethodDef EKFSym_methods[] = {
    {"init_state", (PyCFunction)EKFSym_init_state, METH_VARARGS, ""},
    {"state", (PyCFunction)EKFSym_state, METH_NOARGS, ""},
    {"covs", (PyCFunction)EKFSym_covs, METH_NOARGS, ""},
    {"set_filter_time", (PyCFunction)EKFSym_set_filter_time, METH_VARARGS, ""},
    {"get_filter_time", (PyCFunction)EKFSym_get_filter_time, METH_NOARGS, ""},
    {"set_global", (PyCFunction)EKFSym_set_global, METH_VARARGS, ""},
    {"reset_rewind", (PyCFunction)EKFSym_reset_rewind, METH_NOARGS, ""},
    {"predict", (PyCFunction)EKFSym_predict, METH_VARARGS, ""},
    {"predict_and_update_batch", (PyCFunction)EKFSym_predict_and_update_batch, METH_VARARGS | METH_KEYWORDS, ""},
    {NULL}
};

static PyType_Slot EKFSym_slots[] = {
    {Py_tp_dealloc, (void*)EKFSym_dealloc},
    {Py_tp_init, (void*)EKFSym_init},
    {Py_tp_methods, EKFSym_methods},
    {0, 0}
};

static PyType_Spec EKFSym_spec = {
    "ekf_sym_module.EKFSym",
    sizeof(EKFSymObject),
    0,
    Py_TPFLAGS_DEFAULT,
    EKFSym_slots
};

static PyMethodDef module_methods[] = {
    {NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_ekf_sym_module",
    NULL,
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__ekf_sym_module(void) {
    import_array();

    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;

    PyTypeObject *EKFSymType = (PyTypeObject*)PyType_FromSpec(&EKFSym_spec);
    if (PyModule_AddObject(m, "EKFSym", (PyObject *)EKFSymType) < 0) return NULL;
    Py_INCREF(EKFSymType);

    return m;
}
