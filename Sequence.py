import numpy as np
from ortools.sat.python import cp_model
from Assignment import Assignment
from collections import Counter, defaultdict
from Env import Env
from itertools import permutations

class Pattern:
    def __init__(self):
        self.patterns = {}  # pattern 리스트 저장
    
    def append(self, new_pattern):
        """
        새로운 pattern을 추가
        기존 pattern과 같으면 추가하지 않음
        """        
        # 새로운 패턴의 첫번째 원소가 기존 패턴들의 첫번째 원소와 겹치는지 확인
        if new_pattern[0] not in self.patterns:
            # 겹치지 않으면 새로운 패턴 추가
            self.patterns[new_pattern[0]] = (new_pattern[1], new_pattern[2])
            
        return True
    
    def __len__(self):
        return len(self.patterns)
    
class Guide:
    def __init__(self):
        self.guides = {}  # pattern 리스트 저장
    
    def append(self, new_guide):
        """
        새로운 pattern을 추가
        기존 pattern과 같으면 추가하지 않음
        """
        for guide in self.guides:
            if guide[1] >= new_guide[1] and (self.subset(new_guide[0], guide) or guide[0] == new_guide[0]):
                return False

        # 이제 새거를 반영할거임
        delete_keys = []        
        for guide in self.guides:
            if self.subset(guide, new_guide[0]) and guide[1] <= new_guide[1]:
                delete_keys.append(guide)
            elif guide[0] == new_guide[0] and guide[1] < new_guide[1]:
                delete_keys.append(guide)
        for key in delete_keys: self.guides.pop(key)
                
        self.guides[new_guide[0]] = new_guide[1]
        return True
    
    def subset(self, A, B): # A <= B 인지 확인하는 용도, B가 더 크도록
        if sum(B) >= sum(A): # 두 패턴이 같으면 subset이 아님
            return False
        
        for i, e in enumerate(B):
            if e == 1 and A[i] == 0:
                return False
            
        return True
    
    def __len__(self):
        return len(self.guides)

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, viz):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.viz = viz
        self.__solution_count = 0
        self.first_solution_time = 0
        self.first_solution = 0
        self.obj_values = []
        self.lbd_values = []

    def on_solution_callback(self):
        if self.viz:
            print(
                "Solution %i, time = %f s, objective = %i, lower bound = %i"
                % (self.__solution_count, self.WallTime(), self.ObjectiveValue(), self.best_objective_bound)
            )
        if self.__solution_count == 0: 
            self.first_solution_time = round(self.WallTime(), 2)
            self.first_solution = self.ObjectiveValue()
        self.__solution_count += 1
        self.obj_values.append(self.ObjectiveValue())
        self.lbd_values.append(self.best_objective_bound)

def pattern_to_infomation(machine_num, allocation, pattern: dict):
    # 정보: [False, False, [index], [index, ub, lb], 중복 여부
    pattern_information = [False for _ in range(machine_num)] # default는 처음 들어온 경우, len = 0
    for m in range(machine_num):
        for pi, p in pattern.items():
            pi = np.array(pi)
            if (allocation[:, m] == pi).all():
                if p[0] is not None:
                    if p[0] == p[1]: # 중복되면서 optimal로 풀렸다면 cp 필요 없음, len = 1
                        pattern_information[m] = [tuple(pi)]
                        break
                    else: # 중복되면서 feasible로 풀렸다면 중복되는 패턴의 ub, lb 활용, len = 3
                        pattern_information[m] = [tuple(pi), p[0], p[1]]
                        break
                else: # 중복되면서 unknown으로 풀렸다면 시간 더 주기, len = 2
                    pattern_information[m] = [tuple(pi), None]
    return pattern_information

def check_min_setup(env: Env, families, setup_time=1):
    """
    families: 각 패밀리를 나타내는 dict, 예를 들어 {'a': [0,1,2], 'b': [3,4], 'c': [5,6], ...}
    job_deadlines: job 별 deadline을 담은 dict
    job_durations: job 별 실행시간을 담은 dict
    transition_time: 패밀리 전환 시 대기 시간 (default: 1)
    """
    # 각 패밀리 내의 job들을 deadline 기준으로 미리 정렬
    sorted_families = {family: sorted(jobs, key=lambda job: env.deadline[job])
                         for family, jobs in families.items()}
    
    # 모든 패밀리 순열에 대해 체크
    for order in permutations(sorted_families.keys()):
        time = 0
        valid = True
        # 순서대로 패밀리의 정렬된 job 리스트를 이어 붙여서 수행
        for family in order:
            for job in sorted_families[family]:
                time += env.duration[job]  # job 실행 시간 누적
                if time > env.deadline[job]:
                    # 현재 패밀리 내에서 deadline이 초과되었으므로 break (전체 순열 중단하지 않음)
                    valid = False
                    break
            time += setup_time  # 패밀리 전환 시 대기 시간 추가

        if valid:
            return True

    return False

class Sequence:
    def __init__(self, env, viz=False) -> None:
        self.env = env
        self.viz = viz
        self.count_dict = [{} for _ in range(env.machine_num)]
        self.job_to_family = env.job_to_family
        self.machine_family_num = [0 for _ in range(env.machine_num)]

    def reset(self, env, allocation):
        self.allocation = allocation

        self.machine_jobs = [[] for _ in range(env.machine_num)]
        for machine_index in range(env.machine_num):
            for job_index in range(env.job_num):
                if allocation[job_index, machine_index]:
                    self.machine_jobs[machine_index].append(job_index)

        self.temp_schedule = [[] for _ in range(env.machine_num)]

    def use_EDD(self):
        alloc = Assignment(self.env)
        alloc.EDD(self.allocation)
        return alloc.schedule
    
    def branch(self, m):
        cases = {fam: min(
            (j for j in self.machine_jobs[m] if self.job_to_family[j] == fam 
            and self.env.deadline[j] == min(self.env.deadline[x] for x in self.machine_jobs[m] if self.job_to_family[x] == fam) 
            and self.env.duration[j] == min(self.env.duration[x] for x in self.machine_jobs[m] if self.job_to_family[x] == fam)),
            key=lambda x: self.env.deadline[x],
            default=None
        ) for fam in self.count_dict[m].keys()}
        # None이 아닌 항목과 None인 항목을 분리
        non_none_values = [value for value in cases.values() if value is not None]
        if not non_none_values: # 1. None이 아닌 value가 하나도 없다면
            cases = [None]
        elif len(non_none_values) < len(self.count_dict[m]): # 2. None이 아닌 value가 하나 이상 있다면, 모두가 있지 않는다면
            non_none_keys = tuple(key for key, value in cases.items() if value is not None) # 해당 value들을 모으고, 마지막 원소에 그 key들을 튜플로 추가
            cases = non_none_values + [non_none_keys]
        else: # 모두가 그런 value가 있다면
            cases = non_none_values
        return cases

    def cp_parallel_sequence(self, time_limit, framework_time_limit, framework_elapsed_time, pattern: dict):
        pattern_information = []
        if len(pattern) > 0: # pattern이 있는 경우
            # 정보: [False, False, [index], [index, ub, lb], ...]
            pattern_information = pattern_to_infomation(self.env.machine_num, self.allocation, pattern)
            print(f"pattern selection: {[False if p is False else True for p in pattern_information]}")
        
        # EDD
        self.EDD = self.use_EDD()
        for m in range(self.env.machine_num):
            self.count_dict[m] = Counter(tuple_item[-1] for tuple_item in self.EDD[m])
            self.machine_family_num[m] = len(self.count_dict[m].keys())
            print(f"machine: {m} assigned jobs: {self.machine_jobs[m]}")
        print(f"Family number per machines: {self.machine_family_num}")

        # Main Part
        temp_pattern = [[] for _ in range(self.env.machine_num)]
        schedule = [[] for _ in range(self.env.machine_num)]
        for m in range(self.env.machine_num):
            assigned_job_tuple = tuple([1 if j in self.machine_jobs[m] else 0 for j in range(self.env.job_num)])
            temp_pattern[m].append(assigned_job_tuple) # pattern의 첫번째 정보

            if len(pattern_information) > 0 and pattern_information[m] is not False and len(pattern_information[m]) < 2: # 중복됐으면서 이전에 optimal로 풀렸다면
                pi = pattern_information[m][0]
                ub, lb = pattern[pi][0], pattern[pi][1]
                print("machine:", m, "status: OPTIMAL objective value:", ub, "lower bound:", lb)
                temp_pattern[m].append(ub)
                temp_pattern[m].append(lb)
                continue
    
            else: # 문제를 풀어야 하는 경우
                found_unknown, found_solution = False, False
                ub_hint, lb_hint = None, None

                family_counter = defaultdict(list) 
                for i, b in enumerate(assigned_job_tuple):
                    if b == 1: family_counter[self.env.job_to_family[i]].append(i)
                fam_num = len(family_counter)

                # min setup 확인 -> lb 제공
                if fam_num > 1:
                    if check_min_setup(self.env, family_counter):
                        print("machine:", m, "status: OPTIMAL objective value:", fam_num-1, "lower bound:", fam_num-1)
                        temp_pattern[m].append(fam_num-1)
                        temp_pattern[m].append(fam_num-1)
                        continue
                    else: lb_hint = fam_num

                # 가지치기 가지 제공
                if fam_num > 1: cases = self.branch(m)
                else: cases = [None]
                if len(cases) > 1 and len(self.count_dict[m]) > 1: print(f"machine: {m} Now we use {len(cases)} branchs!")

                # 시간 재조정
                temp_time_limit = time_limit
                if len(pattern_information) > 0 and pattern_information[m] is not False: # 중복된 경우 시간 더 주기
                    temp_time_limit = max((framework_time_limit - framework_elapsed_time)/(len(cases) * (self.env.machine_num - m)), time_limit)
                    print("time!!!", temp_time_limit, (framework_time_limit - framework_elapsed_time)/(len(cases) * (self.env.machine_num - m)))
                    if len(pattern_information[m]) == 3: 
                        ub_hint, lb_hint = pattern_information[m][1], pattern_information[m][2] # 중복됐으면서 이전에 feasible로 풀렸다면

                pre_result = len(self.machine_jobs[m]) # 목적식 값이 가장 작은 경우를 뽑기 위함
                for _, case in enumerate(cases):
                    result = self.cp_sequence_one_machine(m, temp_time_limit, ub_lb=(ub_hint, lb_hint), case=case) # 가능한 많은 시간을 제공
                    if result is None:
                        found_unknown = True
                        continue
                    if result is not False and result[0] < pre_result:
                        pre_result = result[0]
                        temp = result
                        found_solution = True
                        schedule[m] = self.temp_schedule[m]
                        if result[0] == self.machine_family_num[m] - 1:
                            temp = result 
                            break
                if found_solution: 
                    ub, lb = temp # feasible solution이면 p 길이가 3 고정
                    temp_pattern[m].append(ub) 
                    temp_pattern[m].append(lb) # pattern의 두번째 정보
                elif found_unknown: # time over
                    schedule[m] = []
                    temp_pattern[m].append(None)
                else: # infeasible 
                    temp_pattern[m].append(False)

        return temp_pattern, pattern_information, schedule

    def cp_sequence_one_machine(self, machine_index, time_limit, ub_lb=None, case=None):
        # parameters
        job_list = self.machine_jobs[machine_index]
        dummy_job_list = [-1] + job_list
        if len(job_list) == 0:
            return None
        duration = self.env.duration
        deadline = self.env.deadline
        T = sum(duration[j] for j in job_list) + len(job_list) - 1

        model = cp_model.CpModel()

        # Variables
        relation = {}
        setup = {}
        start_time = {}

        for i in job_list:
            start_time[i] = model.NewIntVar(0, T, f'st_{i}')
            for j in job_list:
                relation[i, j] = model.NewBoolVar(f'rel_{i}_{j}')
                setup[i, j] = model.NewBoolVar(f'setup_{i}_{j}')

        for i in job_list:
            relation[i, -1] = model.NewBoolVar(f'rel_{i}_{-1}')
            relation[-1, i] = model.NewBoolVar(f'rel_{-1}_{i}')   

        # objective
        model.minimize(sum([setup[i, j] for i in job_list for j in job_list]))
        
        # optional ub, lb hint
        ub, lb = ub_lb
        if ub is not None: model.add(sum([setup[i, j] for i in job_list for j in job_list]) <= ub)
        if lb is not None: model.add(sum([setup[i, j] for i in job_list for j in job_list]) >= lb)
        
        # deadline
        for j in job_list:
            assert duration[j] <= deadline[j]
            model.add(start_time[j] + duration[j] <= deadline[j])
 
        # Ensure exactly one latest start
        model.add(sum(relation[i, -1] for i in job_list) == 1)

        if case is not None and not isinstance(case, tuple):
            model.add(relation[-1, case] == 1) # 가장 먼저 시작하는 job 고정
        else:
            model.add(sum(relation[-1, i] for i in job_list) == 1) # Ensure exactly one earliest start
            if isinstance(case, tuple):
                temp_ban_job = [i for i in job_list if self.job_to_family[i] in case]
                model.add(sum(relation[-1, i] for i in temp_ban_job) == 0)

        # 각 행의 합이 1이 되도록 제약 조건 추가
        for i in job_list:
            model.add(sum(relation[i, j] for j in dummy_job_list) == 1) # 10, 11

        # 각 열의 합이 1이 되도록 제약 조건 추가
        for j in job_list:
            model.add(sum(relation[i, j] for i in dummy_job_list) == 1) # 7, 8

        # 대각 예외 제약 조건 추가
        for j in job_list:
            model.add(relation[j, j] == 0)
            for i in job_list:
                if j < i:
                    model.add(relation[i, j] + relation[j, i] <= 1)

        # setup 정의
        for i in job_list:
            for j in job_list:
                if self.job_to_family[i] != self.job_to_family[j]:
                    model.add(setup[i, j] >= relation[i, j])
                else:
                    model.add(setup[i, j] == 0)

        # predecessor-successor
        for i in job_list:
            for j in job_list:
                model.add(start_time[i] + duration[i] + setup[i, j] <= start_time[j]).only_enforce_if(relation[i, j])

        # EDD를 힌트로 사용, 더 많은 변수에 값 추가
        if getattr(self, "EDD"): 
            setup_bound = 0
            previous_job = None
            previous_fam = None
            for job, eddst, dur, fam in self.EDD[machine_index]:
                model.add_hint(start_time[job], eddst)
                if previous_job is not None:
                    model.add_hint(relation[previous_job, job], True)
                previous_job = job

                if previous_fam is not None and fam != previous_fam:
                    setup_bound += 1
                previous_fam = fam

            model.add_hint(relation[-1, self.EDD[machine_index][0][0]], True) # for earliest start time
            model.add_hint(relation[self.EDD[machine_index][-1][0], -1], True) # for latest due date

            # LB, UB
            model.add(sum([setup[i, j] for i in job_list for j in job_list]) >= len(self.count_dict[machine_index].keys()) - 1) # trivial lower bound

            model.add(sum([setup[i, j] for i in job_list for j in job_list]) <= setup_bound) # EDD upper bound
            # value_list = sorted(count_dict.values(), reverse=True)
            # total_sum = sum(value_list)
            # largest = value_list[0]
            # if len(value_list) > 1: second_largest = value_list[1]
            # else: second_largest = 0
            # model.add(sum([setup[i, j] for i in job_list for j in job_list]) <= total_sum - largest + second_largest) # trivial upper bound

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solution_printer = SolutionPrinter(self.viz)
        status = solver.solve(model, solution_printer) # solver가 해결하도록

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print("machine:", machine_index, "status:", solver.status_name(status), "objective value:", solver.objective_value,  
            "lower bound:", solver.best_objective_bound, "time:", round(solver.WallTime(), 2))

            start_times = [(j, int(solver.Value(start_time[j])), int(solver.Value(start_time[j])) + int(duration[j]), int(deadline[j]), self.job_to_family[j]) for j in job_list]
            self.temp_schedule[machine_index] = sorted(start_times, key=lambda x: x[1])
            if status == cp_model.OPTIMAL:
                return int(solver.objective_value), int(solver.objective_value)
            return int(solver.objective_value), int(solver.best_objective_bound)
        else:
            print("machine:", machine_index, "status:", solver.status_name(status), "time:", round(solver.WallTime(), 2))
            if status == cp_model.INFEASIBLE:
                return False
            else:
                return None
      
def global_cp(env: Env, Gamma, Lambda, time_limit, viz=True):
    # parameters
    job_num = env.job_num
    machine_num = env.machine_num
    duration = env.duration
    deadline = env.deadline
    T = sum(duration[j] for j in range(job_num)) + job_num - 1 - min(duration)
    SPEC_CDF = env.spec_cdf
    SCALED_V = env.scaled_v

    model = cp_model.CpModel()

    # Variables
    relation = {}
    setup = {}
    start_time = {}
    allocation = {}

    for i in range(job_num):
        start_time[i] = model.NewIntVar(0, T, f'st_{i}')
        for j in range(job_num):
            relation[i, j] = model.NewBoolVar(f'rel_{i}_{j}')
            setup[i, j] = model.NewBoolVar(f'setup_{i}_{j}')
        for m in range(machine_num):
            allocation[i, m] = model.NewBoolVar(f'alloc_{i}_{m}')

    for i in range(job_num):
        relation[-1, i] = model.NewBoolVar(f'rel_{-1}_{i}')
        relation[i, -1] = model.NewBoolVar(f'rel_{i}_{-1}')
        
    # objective
    model.minimize(sum([setup[i, j] for i in range(job_num) for j in range(job_num)]))

    # deadline
    for j in range(job_num):
        assert duration[j] <= deadline[j]
        model.add(start_time[j] + duration[j] <= deadline[j]) # 5

    # allocation
    for i in range(job_num):
        model.add(sum(allocation[i, m] for m in range(machine_num)) == 1) # 1

    # Ensure exactly machine_num earliest start
    model.add(sum(relation[-1, i] for i in range(job_num)) == machine_num) # 17

    # Ensure exactly machine_num latest start
    model.add(sum(relation[i, -1] for i in range(job_num)) == machine_num) # 18

    # 각 행의 합이 1이 되도록 제약 조건 추가
    for i in range(job_num):
        model.add(sum(relation[i, j] for j in range(-1, job_num)) == 1) # 10, 11

    # 각 열의 합이 1이 되도록 제약 조건 추가
    for j in range(job_num):
        model.add(sum(relation[i, j] for i in range(-1, job_num)) == 1) # 7, 8

    # 대각 예외 제약 조건 추가
    for j in range(job_num):
        model.add(relation[j, j] == 0) # 12
        for i in range(job_num):
            if j < i:
                model.add(relation[i, j] + relation[j, i] <= 1) # 13

    # setup 정의
    for i in range(job_num):
        for j in range(job_num):
            if env.job_to_family[i] != env.job_to_family[j]:
                model.add(setup[i, j] >= relation[i, j]) # 15
            else:
                model.add(setup[i, j] == 0) # 16
            if i != j:
                for m in range(machine_num):
                    condition = model.NewBoolVar(f"temp_{i}_{j}_{m}")

                    model.add(condition <= (allocation[i, m] + allocation[j, m]))
                    model.add(condition <= 1 - (allocation[i, m] + allocation[j, m] - 1))
                    model.add(condition >= (allocation[i, m] - allocation[j, m]))
                    model.add(condition >= (allocation[j, m] - allocation[i, m]))

                    model.add(relation[i, j] == 0).only_enforce_if(condition) # 4
                    model.add(relation[j, i] == 0).only_enforce_if(condition) # 4        

    # predecessor-successor
    for i in range(job_num):  
        for j in range(job_num):
            model.add(start_time[i] + duration[i] + setup[i, j] <= start_time[j]).only_enforce_if(relation[i, j]) # 14

    # Constraint: Expected yield
    SCALING_FACTOR = 1000
    model.add(
        sum(
            duration[j] * sum(allocation[j, m] * int(SPEC_CDF[j][m] * SCALING_FACTOR) for m in range(machine_num))
            for j in range(job_num)
        ) >= int(sum(duration) * Gamma * SCALING_FACTOR),
    ) # 2

    # Constraint: Expected variance
    model.add(
        sum(
            duration[j] * sum(allocation[j, m] * int(SCALED_V[j][m] * SCALING_FACTOR) for m in range(machine_num))
            for j in range(job_num)
        ) <= int(sum(duration) * Lambda * SCALING_FACTOR),
    ) # 3

    solver = cp_model.CpSolver()
    solution_printer = SolutionPrinter(viz)
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model, solution_printer) # solver가 해결하도록

    print("status:", solver.status_name(status), "objective value:", solver.objective_value, 
"lower bound:", solver.best_objective_bound, "time:", round(solver.WallTime(), 2))
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if status == cp_model.OPTIMAL:
            return solver.objective_value, round(solver.WallTime(), 2), True, solution_printer.first_solution, solution_printer.first_solution_time
        else:
            return solver.objective_value, round(solver.WallTime(), 2), -1, solution_printer.first_solution, solution_printer.first_solution_time
    elif status == cp_model.INFEASIBLE:
        return False, round(solver.WallTime(), 2), -1, -1, -1
    else:
        print("time limit over")
        return solver.objective_value, round(solver.WallTime(), 2), -1, solution_printer.first_solution, solution_printer.first_solution_time
        