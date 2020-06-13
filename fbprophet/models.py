# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
from abc import abstractmethod, ABC
from typing import Tuple
from collections import OrderedDict
from enum import Enum
import pickle
import pkg_resources
import numpy as np
import os


class IStanBackend(ABC):
    def __init__(self, logger, constr_regressors):
        self.model = self.load_model(constr_regressors)
        self.logger = logger

    @staticmethod
    @abstractmethod
    def get_type():
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def fit(self, stan_init, stan_data, **kwargs) -> dict:
        pass

    @abstractmethod
    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def build_model(target_dir, model_dir):
        pass


class CmdStanPyBackend(IStanBackend):

    @staticmethod
    def get_type():
        return StanBackendEnum.CMDSTANPY.name

    @staticmethod
    def build_model(target_dir, model_dir, model_name, target_name):
        from shutil import copy
        import cmdstanpy
        sm = cmdstanpy.Model(stan_file=os.path.join(os.sep.join(model_dir), model_name))
        sm.compile()
        copy(sm.exe_file, os.path.join(os.sep.join(target_dir), target_name))

    def load_model(self, constr_regressors):
        import cmdstanpy

        models_dict = {False:  'prophet_model.bin',
                       True:  os.path.join('contrib', 'prophet_normal_truncated.bin'),
                       }
        cur_model = models_dict[bool(constr_regressors)]
        model_file = pkg_resources.resource_filename(
            'fbprophet',
            f'stan_model/{cur_model}',
        )

        model_dir = model_file.split(os.sep)[:-1]
        model_name = model_file.split(os.sep)[-1]
        if model_name not in os.listdir(os.sep.join(model_dir)):
            target_name, model_name = model_name, model_name.replace('.bin', '.stan')
            target_dir, model_dir = model_dir, model_dir
            print(f'Building model: {model_name}\nModel dir: {os.sep.join(model_dir)}\nName: {model_name}')
            self.build_model(target_dir, model_dir, model_name, target_name)

        return cmdstanpy.Model(exe_file=model_file)

    def fit(self, stan_init, stan_data, **kwargs):
        (stan_init, stan_data) = self.prepare_data(stan_init, stan_data)


        if 'algorithm' not in kwargs:
            kwargs['algorithm'] = 'Newton' if stan_data['T'] < 100 else 'LBFGS'
        iterations = int(1e4)
        try:
            stan_fit = self.model.optimize(data=stan_data,
                                           inits=stan_init,
                                           iter=iterations,
                                           **kwargs)
        except RuntimeError as e:
            # Fall back on Newton
            if kwargs['algorithm'] != 'Newton':
                self.logger.warning(
                    'Optimization terminated abnormally. Falling back to Newton.'
                )
                kwargs['algorithm'] = 'Newton'
                stan_fit = self.model.optimize(data=stan_data,
                                               inits=stan_init,
                                               iter=iterations,
                                               **kwargs)
            else:
                raise e

        params = self.stan_to_dict_numpy(stan_fit.column_names, stan_fit.optimized_params_np)
        for par in params:
            params[par] = params[par].reshape((1, -1))
        return params

    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:
        (stan_init, stan_data) = self.prepare_data(stan_init, stan_data)

        if 'chains' not in kwargs:
            kwargs['chains'] = 4
        if 'warmup_iters' not in kwargs:
            kwargs['warmup_iters'] = samples // 2
        stan_fit = self.model.sample(data=stan_data,
                                     inits=stan_init,
                                     sampling_iters=samples,
                                     **kwargs)
        res = stan_fit.sample
        (samples, c, columns) = res.shape
        res = res.reshape((samples * c, columns))
        params = self.stan_to_dict_numpy(stan_fit.column_names, res)

        for par in params:
            s = params[par].shape
            if s[1] == 1:
                params[par] = params[par].reshape((s[0],))

            if par in ['delta', 'beta'] and len(s) < 2:
                params[par] = params[par].reshape((-1, 1))

        return params, stan_fit

    @staticmethod
    def prepare_data(init, data) -> Tuple[dict, dict]:
        cmdstanpy_data = {
            'T': data['T'],
            'S': data['S'],
            'K': data['K'],
            'tau': data['tau'],
            'trend_indicator': data['trend_indicator'],
            'y': data['y'].tolist(),
            't': data['t'].tolist(),
            'cap': data['cap'].tolist(),
            't_change': data['t_change'].tolist(),
            's_a': data['s_a'].tolist(),
            's_m': data['s_m'].tolist(),
            'X': data['X'].to_numpy().tolist(),
            'sigmas': data['sigmas'],
            'mus': data['mus'],
        }

        if 'constr_vec' in list(data.keys()):
            cmdstanpy_data.update({'n_constr': data['n_constr'],
                                   'constr_vec': data['constr_vec'],
                                   'norm_vec': [int(x) for x in range(1, data['K']+1) if x not in data['constr_vec']],
                                   'B': data['B'],
                                   })

        cmdstanpy_init = {
            'k': init['k'],
            'm': init['m'],
            'delta': init['delta'].tolist(),
            'beta': init['beta'].tolist(),
            'sigma_obs': 1
        }
        return (cmdstanpy_init, cmdstanpy_data)

    @staticmethod
    def stan_to_dict_numpy(column_names: Tuple[str, ...], data: np.array):
        output = OrderedDict()

        prev = None

        start = 0
        end = 0
        two_dims = True if len(data.shape) > 1 else False
        for cname in column_names:
            parsed = cname.split(".")

            curr = parsed[0]
            if prev is None:
                prev = curr

            if curr != prev:
                if prev in output:
                    raise RuntimeError(
                        "Found repeated column name"
                    )
                if two_dims:
                    output[prev] = np.array(data[:, start:end])
                else:
                    output[prev] = np.array(data[start:end])
                prev = curr
                start = end
                end += 1
            else:
                end += 1

        if prev in output:
            raise RuntimeError(
                "Found repeated column name"
            )
        if two_dims:
            output[prev] = np.array(data[:, start:end])
        else:
            output[prev] = np.array(data[start:end])
        return output


class PyStanBackend(IStanBackend):

    @staticmethod
    def get_type():
        return StanBackendEnum.PYSTAN.name

    @staticmethod
    def build_model(model_dir, model_name):
        import pystan

        target_name, model_name = model_name, model_name.replace('.pkl', '.stan')
        target_dir, model_dir = model_dir, model_dir

        with open(os.path.join(model_dir, model_name)) as f:
            model_code = f.read()
        sm = pystan.StanModel(model_code=model_code)

        with open(os.path.join(target_dir, target_name), 'wb') as f:
            pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)

    def sampling(self, stan_init, stan_data, samples, **kwargs) -> dict:

        if 'constr_vec' in stan_data.keys():
            stan_data.update({'norm_vec':
                                  [int(x) for x in range(1, stan_data['K'] + 1) if x not in stan_data['constr_vec']],
                              }
                             )
        args = dict(
            data=stan_data,
            init=lambda: stan_init,
            iter=samples,
        )
        args.update(kwargs)
        stan_fit = self.model.sampling(**args)
        out = dict()
        for par in stan_fit.model_pars:
            out[par] = stan_fit[par]
            # Shape vector parameters
            if par in ['delta', 'beta'] and len(out[par].shape) < 2:
                out[par] = out[par].reshape((-1, 1))
        return out, stan_fit

    def fit(self, stan_init, stan_data, **kwargs) -> dict:

        args = dict(
            data=stan_data,
            init=lambda: stan_init,
            algorithm='Newton' if stan_data['T'] < 100 else 'LBFGS',
            iter=1e4,
        )
        args.update(kwargs)
        try:
            params = self.model.optimizing(**args)
        except RuntimeError:
            # Fall back on Newton
            self.logger.warning(
                'Optimization terminated abnormally. Falling back to Newton.'
            )
            args['algorithm'] = 'Newton'
            params = self.model.optimizing(**args)

        for par in params:
            params[par] = params[par].reshape((1, -1))

        return params

    def load_model(self, constr_regressors):
        """Load compiled Stan model"""

        models_dict = {False:  'prophet_model.pkl',
                       True:  os.path.join('contrib', 'prophet_normal_truncated.pkl'),
                       }
        cur_model = models_dict[bool(constr_regressors)]
        model_file = pkg_resources.resource_filename(
                'fbprophet',
                f'stan_model/{cur_model}',
        )
        print(model_file)
        model_dir = model_file.split(os.sep)[:-1]
        model_name = model_file.split(os.sep)[-1]
        if model_name not in os.listdir('/'.join(model_dir)):
            print(f'Building model: {model_name}\nModel dir: {os.sep.join(model_dir)}\nName: {model_name}')
            self.build_model(os.sep.join(model_dir), model_name)

        with open(model_file, 'rb') as f:
            return pickle.load(f)


class StanBackendEnum(Enum):
    PYSTAN = PyStanBackend
    CMDSTANPY = CmdStanPyBackend

    @staticmethod
    def get_backend_class(name: str) -> IStanBackend:
        try:
            return StanBackendEnum[name].value
        except KeyError:
            raise ValueError("Unknown stan backend: {}".format(name))
