# _*_coding:utf-8_*_

# import tensorflow as tf
import gc
from utils import config, DUMPS_utlis
import pickle
from _others.fid.lidargen_fid import get_fid
from _lib.queue.seed import Seed


class Fuzzer(object):
    """Class representing the fuzzer itself."""

    def __init__(self,
                 queue,
                 metadata_function,
                 objective_function,
                 mutation_function,
                 fetch_function,
                 iterate_function,
                 frd_function,
                 plot=True):
        '''
        Initializes the fuzzer object.
        Args:
            queue: The queue to use for managing the fuzzing process.
            objective_function: A function to build objective.
            mutation_function: A function to mutate the data.
            fetch_function: A function to fetch prediction for fuzzing.
            iterate_function: A function to iterate over the fuzzed data.
            frd_function: A function for the calculation of FRD score.
        '''

        self.plot = plot
        self.queue = queue
        self.metadata_function = metadata_function
        self.objective_function = objective_function
        self.mutation_function = mutation_function
        self.fetch_function = fetch_function
        self.iterate_function = iterate_function
        self.frd_function = frd_function

    def save(self, iteration):
        cov1 = self.queue.DUMPS['coverage1']
        cov2 = self.queue.DUMPS['coverage2']
        err_cov1 = self.queue.DUMPS['error_coverage1']
        err_cov2 = self.queue.DUMPS['error_coverage2']

        print(
            f'Coverage1 : {cov1:.02%}\nCoverage2 : {cov2:.02%}\nError Coverage1 : {err_cov1:.02%}\nError Coverage2 : {err_cov2:.02%}'
        )
        print('\n')

        fn = f'{self.queue.out_dir}/result/result_{iteration}.pickle'
        with open(fn, 'wb') as f:
            pickle.dump(self.queue.DUMPS, f)

    def loop(self, iterations):
        """Fuzzes a machine learning model in a loop, making *iterations* steps."""
        iteration = 0
        while True:

            if len(self.queue.queue) < 1 or iteration >= iterations:
                break

            if True:
                gc.collect()

            selected_seed_list = []
            mutated_data_dict = []

            for __i in range(4):
                method, parent, data_dict = self.mutation_function()
                print(' -- ok')

                seed = Seed(parent.root_seed, parent)

                selected_seed_list.append(seed)
                mutated_data_dict.append(data_dict)

                # calculate frd
                if method in config.scene_level:
                    seed.cal_frd_with = None
                    mutated_data_dict[-1]['frd_score'] = self.queue.frd_score[
                        f'{parent.id:06d}']
                    print('  ignore calculation of frd.')
                elif method in config.object_level:
                    ref_batch = pickle.load(
                        open(self.queue.frd_fname[f'{seed.cal_frd_with:06d}'],
                             'rb'))

                    while isinstance(ref_batch, list):
                        ref_batch = ref_batch[0]

                    frd_score = get_fid(
                        self.frd_function(ref_batch['points']),
                        self.frd_function(mutated_data_dict[-1]['points']),
                    )

                    frd_score += self.queue.frd_score[
                        f'{seed.cal_frd_with:06d}']

                    mutated_data_dict[-1]['frd_score'] = frd_score
                else:
                    raise NotImplementedError

                after_score = mutated_data_dict[-1]['frd_score']
                if seed.cal_frd_with is None:
                    before_score = after_score
                else:
                    before_score = self.queue.frd_score[
                        f'{seed.cal_frd_with:06d}']

                frd_limit = self.queue.DUMPS['frd_limit']
                print(
                    f'  FRD : {before_score} --> {after_score}, limit : {frd_limit}'
                )
                if after_score > frd_limit:
                    selected_seed_list.pop()
                    mutated_data_dict.pop()
                    print(
                        f'  -- Exceeded FRD Score Limit ## Current Score : {after_score}, Limit : {frd_limit}'
                    )

            if len(mutated_data_dict) > 0:
                pred_boxes, pred_scores, pred_labels = self.fetch_function(
                    mutated_data_dict)

                for _b, data_dict in enumerate(mutated_data_dict):
                    data_dict['pred_boxes'] = pred_boxes[_b]
                    data_dict['pred_scores'] = pred_scores[_b]
                    data_dict['pred_labels'] = pred_labels[_b]

            if len(mutated_data_dict) > 0:
                self.iterate_function(self.queue, selected_seed_list,
                                      mutated_data_dict,
                                      self.objective_function)
            else:
                print(f'## All Exceeded FRD Score Limit ##')

                DUMPS_utlis.updateIter_DUMPS_errorMetric(self.queue.DUMPS)

            iteration += 1

            print('\n')

            DUMPS_utlis.updateCoverage_DUMPS(self.queue.DUMPS)

            if self.queue.check_point != 0:
                if (iteration) % self.queue.check_point == 0:
                    self.save(iteration)
                elif iteration == iterations:
                    self.save(iteration)
            else:
                self.save(iteration)

        return None
