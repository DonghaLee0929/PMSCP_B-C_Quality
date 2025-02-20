# =================================================================================================================== #    
    # Environment class
    # def _calculate_cvar(self, alpha):
    #     return self.mean - self.std * (norm.pdf(norm.ppf(1 - alpha)) / alpha)

    # def ublambda(self, gamma, alpha):
    #     temp = (self.duration**2) / sum(self.duration) * (self.mean.max(axis=1)/self.spec - alpha)
    #     return (1/(1-alpha)) * (gamma + sum(temp))
    
    # def approxlambda(self, gamma):
    #     temp = (self.duration**3) / sum(self.duration) * self.mean.max(axis=1) / (self.spec*(self.duration-gamma))
    #     return sum(temp)
    
    # def reset(self, seed=42, alpha=0.7, tight:str='high', quality:str='high'):
    #     # Initialize random generator with seed
    #     rng = np.random.default_rng(seed)

    #     # 최소 1개의 job을 모든 family에 먼저 할당
    #     min_jobs_per_family = 1
    #     remaining_jobs = self.job_num - min_jobs_per_family * self.family_num
    #     # 파레토 분포 활용, a가 작을수록 편차가 심함
    #     raw_sizes = rng.pareto(a=1, size=self.family_num) + 1
    #     raw_sizes /= raw_sizes.sum()  # 정규화
    #     additional_jobs = np.round(raw_sizes * remaining_jobs).astype(int)
    #     # 각 family에 최소 1개 + 추가 job 배정
    #     family_sizes = additional_jobs + min_jobs_per_family
    #     # 총합 조정 (오차 보정)
    #     adjustment = self.job_num - np.sum(family_sizes)
    #     family_sizes[0] += adjustment
    #     # 결과 생성
    #     self.family = []
    #     start = 0
    #     for size in family_sizes:
    #         self.family.append(list(range(start, start + size)))
    #         start += size

    #     self.job_to_family = {}
    #     for family_index, jobs in enumerate(self.family):
    #         # print(len(jobs), end=" ")
    #         for job in jobs:
    #             self.job_to_family[job] = family_index

    #     # Initialize deadlines array
    #     self.deadline = np.zeros(self.job_num, dtype=int)

    #     # Use the random generator for all random values
    #     self.duration = np.ones(self.job_num, dtype=int)

    #     # Calculate distribution parameters
    #     a = 5
    #     b_a = 25

    #     # Step 2: Assign deadlines for each job
    #     for f in range(self.family_num):
    #         temp = self._deadline_distribution(rng, tight, a, b_a * 1)
    #         for i in self.family[f]:
    #             self.deadline[i] = temp

    #     # Randomly initialize mean and std
    #     mean, std, spec_a, spec_b = self._quality_bound(quality)
    #     self.spec = rng.uniform(spec_a, spec_b, size=self.job_num)
    #     self.mean = np.full((self.job_num, self.machine_num), mean)
    #     self.std = np.full((self.job_num, self.machine_num), std)

    #     # Calculate cdf, scaled_v, and scaled_cvar
    #     self.spec_cdf = np.round(norm.cdf(self.spec[:, np.newaxis], loc=self.mean, scale=self.std), decimals=3)
    #     self.scaled_v = np.round(self._calculate_conditional_means(self.mean, self.std, self.spec[:, np.newaxis]) / self.spec[:, np.newaxis], decimals=3)
    #     self.scaled_cvar = self._calculate_cvar(alpha) / self.spec[:, np.newaxis]

    #     self._create_state()
    #     return self.env    

# =================================================================================================================== # 
    # Assign class
    # def gurobi_allocation_feasibility(self, Gamma, Lambda, time_limit, cvar=False, all_bans=False): # 수정 예정
    #     # parameters
    #     job_num = self.env.job_num
    #     machine_num = self.env.machine_num
    #     family_num = self.env.family_num
    #     family = self.env.family
    #     duration = self.env.duration
    #     spec_cdf = self.env.spec_cdf
    #     if cvar:
    #         scaled_v = self.env.scaled_cvar
    #     else:
    #         scaled_v = self.env.scaled_v

    #     model = Model("Allocation")
    #     model.setParam('OutputFlag', self.viz)
    #     model.setParam("TimeLimit", time_limit)  # 시간 제한 설정

    #     # Variables
    #     allocations = model.addVars(job_num, machine_num, vtype=GRB.BINARY, name="allocations")

    #     # Objective: Minimize c_max
    #     model.setObjective(1, GRB.MINIMIZE)

    #     # Constraint: Job uniqueness
    #     for j in range(job_num):
    #         model.addConstr(
    #             quicksum(allocations[j, m] for m in range(machine_num)) == 1,
    #             name=f"job_uniqueness_{j}"
    #         )

    #     # Optional Constraint: Ban list
    #     if not all_bans:
    #         for m in range(machine_num):
    #             for i, j in self.ban_list[m]:
    #                 model.addConstr(allocations[i, m] + allocations[j, m] <= 1,
    #                     name=f"ban_list_{j}"
    #                 )
    #     else:
    #         for bans_per_machine in self.ban_list:
    #             for m in range(machine_num):
    #                 model.addConstr(
    #                     quicksum(allocations[i, m] for i in bans_per_machine) <= len(bans_per_machine) - 1,
    #                         name=f"ban_list_{m}"
    #                 )

    #     # Constraint: Expected yield
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * spec_cdf[j][m] for m in range(machine_num))
    #             for j in range(job_num)
    #         ) >= sum(duration) * Gamma,
    #         name="expected_yield"
    #     )

    #     # Constraint: Expected variance
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * scaled_v[j][m] for m in range(machine_num))
    #             for j in range(job_num)
    #         ) <= sum(duration) * Lambda,
    #         name="expected_variance"
    #     )

    #     # Solve the model
    #     model.optimize()

    #     # Check the solution
    #     self.time = model.Runtime
    #     self.status = model.Status
    #     if model.Status == GRB.INFEASIBLE or model.Status == GRB.UNBOUNDED:
    #         return False
        
    #     self.allocation = np.zeros((job_num, machine_num), dtype=bool)
    #         # Fill the numpy array with variable values
    #     for j in range(job_num):
    #         for m in range(machine_num):
    #             self.allocation[j, m] = bool(allocations[j, m].X)  # Extract the value of allocations[j, m]
                
    #     if model.Status == GRB.OPTIMAL:
    #         return True
    #     return True

    # def gurobi_allocation_makespan(self, Gamma, Lambda, time_limit):
    #     # parameters
    #     job_num = self.env.job_num
    #     machine_num = self.env.machine_num
    #     family_num = self.env.family_num
    #     family = self.env.family
    #     duration = self.env.duration
    #     spec_cdf = self.env.spec_cdf
    #     scaled_v = self.env.scaled_v

    #     model = Model("Allocation")
    #     model.setParam('OutputFlag', self.viz)
    #     model.setParam("TimeLimit", time_limit)  # 시간 제한 설정

    #     # Variables
    #     allocations = model.addVars(job_num, machine_num, vtype=GRB.BINARY, name="allocations")
    #     c_max = model.addVar(lb=0, ub=np.sort(duration)[-job_num:].sum(), vtype=GRB.CONTINUOUS, name="c_max")

    #     # Objective: Minimize c_max
    #     model.setObjective(c_max, GRB.MINIMIZE)

    #     # Constraint: c_max
    #     for m in range(machine_num):
    #         model.addConstr(
    #             quicksum(allocations[j, m] * duration[j] for j in range(job_num)) <= c_max,
    #             name=f"c_max_machine_{m}"
    #         )

    #     # Constraint: Job uniqueness
    #     for j in range(job_num):
    #         model.addConstr(
    #             quicksum(allocations[j, m] for m in range(machine_num)) == 1,
    #             name=f"job_uniqueness_{j}"
    #         )

    #     # Optional Constraint: Ban list
    #     for m in range(machine_num):
    #         for i, j in self.ban_list[m]:
    #             model.addConstr(allocations[i, m] + allocations[j, m] <= 1,
    #                 name=f"ban_list_{j}"
    #             )

    #     # Constraint: Expected yield
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * spec_cdf[j][m] for m in range(machine_num))
    #             for j in range(job_num)
    #         ) >= sum(duration) * Gamma,
    #         name="expected_yield"
    #     )

    #     # Constraint: Expected variance
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * scaled_v[j][m] for m in range(machine_num))
    #             for j in range(job_num)
    #         ) <= sum(duration) * Lambda,
    #         name="expected_variance"
    #     )

    #     # Solve the model
    #     model.optimize()

    #     # Check the solution
    #     self.time = model.Runtime
    #     self.status = model.Status
    #     if model.Status == GRB.INFEASIBLE or model.Status == GRB.UNBOUNDED:
    #         print("No feasible solution found.", round(model.Runtime, 2), "seconds")
    #         return None
        
    #     self.allocation = np.zeros((job_num, machine_num), dtype=bool)
    #         # Fill the numpy array with variable values
    #     for j in range(job_num):
    #         for m in range(machine_num):
    #             self.allocation[j, m] = bool(allocations[j, m].X)  # Extract the value of allocations[j, m]
                
    #     if model.Status == GRB.OPTIMAL:
    #         print(int(model.ObjVal), "Optimal", round(model.Runtime, 2), "seconds")
    #         return int(model.ObjVal)
    #     print(int(model.ObjVal), "Feasible", round(model.Runtime, 2), "seconds")
    #     return int(model.ObjVal)
    
    # def gurobi_allocation_max_setup(self, Gamma, Lambda, time_limit, bigm=False):
    #     # parameters
    #     job_num = self.env.job_num
    #     machine_num = self.env.machine_num
    #     family_num = self.env.family_num
    #     family = self.env.family
    #     duration = self.env.duration
    #     spec_cdf = self.env.spec_cdf
    #     scaled_v = self.env.scaled_v

    #     model = Model("Allocation")
    #     model.setParam('OutputFlag', self.viz)
    #     model.setParam("TimeLimit", time_limit)  # 시간 제한 설정

    #     # Variables
    #     allocations = model.addVars(job_num, machine_num, vtype=GRB.BINARY, name="allocations")
    #     s_max = model.addVar(lb=0, ub=job_num, vtype=GRB.CONTINUOUS, name="c_max")

    #     machine_max_family = model.addVars(machine_num, vtype=GRB.CONTINUOUS, name="machine_max_family")
    #     machine_sub_max_family = model.addVars(machine_num, vtype=GRB.CONTINUOUS, name="machine_sub_max_family")
    #     machine_check_max = model.addVars(machine_num, family_num, vtype=GRB.BINARY, name="machine_check_max")

    #     # Objective: Minimize c_max
    #     model.setObjective(s_max, GRB.MINIMIZE)

    #     # Constraint: s_max
    #     for m in range(machine_num):
    #         model.addConstr(
    #             quicksum(quicksum(allocations[j, m] for j in family[f]) for f in range(family_num)) 
    #                 - machine_max_family[m] + machine_sub_max_family[m] <= s_max,
    #             name=f"s_max_machine_{m}"
    #         )

    #     # Constraint: machine_family_relation
    #     for m in range(machine_num):
    #         model.addConstr(quicksum(machine_check_max[m, f] for f in range(family_num)) == 1, name=f"machine_check_max_{m}")
                
    #     for m in range(machine_num):
    #         model.addConstrs((machine_max_family[m] >= quicksum(allocations[f, m] for f in family[i]) for i in range(family_num)), 
    #                          name=f"machine_max_family_lb_{m}")
    #         model.addConstr(machine_sub_max_family[m] <= machine_max_family[m], name=f"machine_sub_max_family_ub_{m}")
    #         if bigm:
    #             M = 1e3
    #             model.addConstrs((machine_max_family[m] <= quicksum(allocations[f, m] for f in family[i]) + M * (1 - machine_check_max[m, i]) for i in range(family_num)), 
    #                             name=f"machine_max_family_ub_{m}")
    #             model.addConstrs((machine_sub_max_family[m] >= quicksum(allocations[f, m] for f in family[i]) - M * machine_check_max[m, i] for i in range(family_num)), 
    #                             name=f"machine_sub_max_family_lb_{m}")
    #         else:
    #             for i in range(family_num):
    #                 model.addGenConstrIndicator(machine_check_max[m, i], True, machine_max_family[m] <= quicksum(allocations[f, m] for f in family[i]), name=f"machine_max_family_ub_{m}{i}")
    #                 model.addGenConstrIndicator(machine_check_max[m, i], False, machine_sub_max_family[m] >= quicksum(allocations[f, m] for f in family[i]), name=f"machine_sub_max_family_lb_{m}{i}")
                
            
    #     # Constraint: Job uniqueness
    #     for j in range(job_num):
    #         model.addConstr(
    #             quicksum(allocations[j, m] for m in range(machine_num)) == 1,
    #             name=f"job_uniqueness_{j}"
    #         )

    #     # Optional Constraint: Ban list
    #     for m in range(machine_num):
    #         for i, j in self.ban_list[m]:
    #             model.addConstr(allocations[i, m] + allocations[j, m] <= 1,
    #                 name=f"ban_list_{j}"
    #             )

    #     # Constraint: Expected yield
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * spec_cdf[j][m] for m in range(machine_num))
    #             for j in range(job_num)
    #         ) >= sum(duration) * Gamma,
    #         name="expected_yield"
    #     )

    #     # Constraint: Expected variance
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * scaled_v[j][m] for m in range(machine_num))
    #             for j in range(job_num)
    #         ) <= sum(duration) * Lambda,
    #         name="expected_variance"
    #     )

    #     # Solve the model
    #     model.optimize()

    #     # Check the solution
    #     self.time = model.Runtime
    #     self.status = model.Status
    #     if model.Status == GRB.INFEASIBLE or model.Status == GRB.UNBOUNDED:
    #         print("No feasible solution found.", round(model.Runtime, 2), "seconds")
    #         return None
        
    #     self.allocation = np.zeros((job_num, machine_num), dtype=bool)
    #         # Fill the numpy array with variable values
    #     for j in range(job_num):
    #         for m in range(machine_num):
    #             self.allocation[j, m] = bool(allocations[j, m].X)  # Extract the value of allocations[j, m]
                
    #     if model.Status == GRB.OPTIMAL:
    #         print(int(model.ObjVal), "Optimal", round(model.Runtime, 2), "seconds")
    #         return int(model.ObjVal)
    #     print(int(model.ObjVal), "Feasible", round(model.Runtime, 2), "seconds")
    #     return int(model.ObjVal)

    # def cp_allocation_min_setup(self, Gamma, Lambda, time_limit):
    #     # Parameters from environment
    #     job_num = self.env.job_num
    #     machine_num = self.env.machine_num
    #     family_num = self.env.family_num
    #     duration = self.env.duration
    #     deadline = self.env.deadline
    #     family = self.env.family
    #     spec_cdf = self.env.spec_cdf
    #     scaled_v = self.env.scaled_v

    #     # Create CP model
    #     model = cp_model.CpModel()
        
    #     # Variables
    #     allocation = {}
    #     machine_family = {}
    #     s_min = []

    #     for j in range(job_num):
    #         for m in range(machine_num):
    #             allocation[j, m] = model.NewBoolVar(f'alloc_{j}_{m}')

    #     for m in range(machine_num):
    #         s_min.append(model.NewIntVar(0, family_num - 1, f's_min_{m}'))
    #         for f in range(family_num):
    #             machine_family[m, f] = model.NewBoolVar(f'machine_family_{m}_{f}')

    #     # Objective: Minimize setup counts
    #     model.Minimize(sum(s_min))

    #     # Constraint: Family-based setup counts
    #     for m in range(machine_num):
    #         model.Add(sum(machine_family[m, f] for f in range(family_num)) - 1 <= s_min[m])

    #     # Constraint: machine_family relation
    #     for m in range(machine_num):
    #         for f in range(family_num):
    #             job_list = family[f]
    #             model.Add(sum(allocation[j, m] for j in job_list) >= 1).OnlyEnforceIf(machine_family[m, f])
    #             model.Add(sum(allocation[j, m] for j in job_list) == 0).OnlyEnforceIf(machine_family[m, f].Not())

    #     # Job uniqueness constraint
    #     for j in range(job_num):
    #         model.Add(sum(allocation[j, m] for m in range(machine_num)) == 1)

    #     # Machine capacity constraint
    #     for m in range(machine_num):
    #         model.Add(
    #             sum(allocation[j, m] * duration[j] for j in range(job_num)) + sum(machine_family[m, f] for f in range(family_num)) - 1
    #             <= max(deadline)
    #         )

    #         for f in range(family_num):
    #             model.Add(
    #                 sum(allocation[j, m] * duration[j] for j in family[f]) <= max([deadline[j] for j in family[f]])
    #             )

    #     # Yield constraint
    #     scaling_factor = 1000
    #     model.Add(
    #         sum(
    #             duration[j] * sum(allocation[j, m] * int(spec_cdf[j][m] * scaling_factor) for m in range(machine_num))
    #             for j in range(job_num)
    #         ) >= int(sum(duration) * Gamma * scaling_factor)
    #     )

    #     # Variance constraint
    #     model.Add(
    #         sum(
    #             duration[j] * sum(allocation[j, m] * int(scaled_v[j][m] * scaling_factor) for m in range(machine_num))
    #             for j in range(job_num)
    #         ) <= int(sum(duration) * Lambda * scaling_factor)
    #     )

    #     # Ban list constraint
    #     for bitmask, sum_ban in self.ban_list.bans.items():
    #         ban = self.ban_list.to_list(bitmask, job_num)
    #         for m in range(machine_num):
    #             model.Add(
    #                 sum(allocation[j, m] * ban[j] for j in range(job_num)) <= sum_ban - 1
    #             )

    #     # Solver setup
    #     solver = cp_model.CpSolver()
    #     solver.parameters.max_time_in_seconds = time_limit
    #     self.status = solver.Solve(model)
    #     self.time = round(solver.WallTime(), 2)

    #     print("Time =", self.time, "Status:", solver.StatusName(self.status))
    #     if self.status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    #         self.allocation = np.zeros((job_num, machine_num), dtype=bool)
    #         # Fill the numpy array with variable values
    #         for j in range(job_num):
    #             for m in range(machine_num):
    #                 try:
    #                     self.allocation[j, m] = bool(solver.Value(allocation[j, m]))  # Extract the value of allocations[j, m]
    #                 except:
    #                     return None
            
    #         return round(solver.ObjectiveValue())
    #     elif self.status == cp_model.INFEASIBLE:
    #         return False
    #     else:
    #         return None

# =================================================================================================================== #    
    # Sequence class
    # def cp_parallel_sequence(self, time_limit, framework_time_limit, framework_start_time, pattern: dict):
    #     pattern_information = []
    #     if len(pattern) > 0: # pattern이 있는 경우
    #         # 정보: [False, False, [index], [index, (ub, lb)], ...]
    #         pattern_information = pattern_to_infomation(self.env.machine_num, self.allocation, pattern)
    #         print(f"pattern selection: {[False if p is False else True for p in pattern_information]}")
        
    #     # EDD
    #     self.EDD = self.use_EDD()
    #     for m in range(self.env.machine_num):
    #         self.count_dict[m] = Counter(tuple_item[-1] for tuple_item in self.EDD[m])
    #         self.machine_family_num[m] = len(self.count_dict[m].keys())
    #         print(f"machine: {m} assigned jobs: {self.machine_jobs[m]}")
    #     print(f"Family number per machines: {self.machine_family_num}")

    #     # Main Part
    #     temp_pattern = [[] for _ in range(self.env.machine_num)]
    #     schedule = [[] for _ in range(self.env.machine_num)]
    #     for m in range(self.env.machine_num):
    #         temp_pattern[m].append(tuple([1 if j in self.machine_jobs[m] else 0 for j in range(self.env.job_num)])) # pattern의 첫번째 정보
    #         cases = self.branch(m)
    #         found_unknown, found_solution = False, False

    #         if len(pattern_information) == 0 or pattern_information[m] is False: # pattern을 넣어주지 않는 경우와 처음 들어온 pattern인 경우
                
    #             if len(cases) > 1 and len(self.count_dict[m]) > 1: print(f"machine: {m} Now we use {len(cases)} branchs!")
    #             pre_result = len(self.machine_jobs[m]) # 목적식 값이 가장 작은 경우를 뽑기 위함
    #             for tt, case in enumerate(cases):
    #                 result = self.improved_cp_sequence_one_machine(m, time_limit, case=case)
    #                 if result is None:
    #                     found_unknown = True
    #                     continue
    #                 if result is not False and result[0] < pre_result:
    #                     pre_result = result[0]
    #                     temp = result
    #                     found_solution = True
    #                     schedule[m] = self.temp_schedule[m]
    #                     if result[0] == self.machine_family_num[m] - 1:
    #                         temp = result 
    #                         break
                
    #         else: # pattern을 넣었고 중복되는 경우

    #             if len(pattern_information[m]) >= 2: # 중복됐으면서 이전에 feasible/unknown로 풀렸다면
    #                 ub_lb = None
    #                 if len(pattern_information[m]) == 3: ub_lb = (pattern_information[m][1], pattern_information[m][2])

    #                 if len(cases) > 1 and len(self.count_dict[m]) > 1: print(f"machine: {m} Now we use {len(cases)} branchs!")
    #                 pre_result = len(self.machine_jobs[m]) # 목적식 값이 가장 작은 경우를 뽑기 위함
    #                 for tt, case in enumerate(cases):
    #                     temp_time_limit = max((framework_time_limit - framework_start_time)/(len(cases) * (self.env.machine_num - m)), time_limit)
    #                     result = self.improved_cp_sequence_one_machine(m, temp_time_limit, ub_lb=ub_lb, case=case) # 가능한 많은 시간을 제공
    #                     if result is None:
    #                         found_unknown = True
    #                         continue
    #                     if result is not False and result[0] < pre_result:
    #                         pre_result = result[0]
    #                         temp = result
    #                         found_solution = True
    #                         schedule[m] = self.temp_schedule[m]
    #                         if result[0] == self.machine_family_num[m] - 1:
    #                             temp = result 
    #                             break

    #             else: # 중복됐으면서 이전에 optimal로 풀렸다면
    #                 pi = pattern_information[m][0]
    #                 print("machine:", m, "status: OPTIMAL objective value:", pattern[pi][0], "lower bound:", pattern[pi][1])
    #                 temp_pattern[m].append(pattern[pi][0])
    #                 temp_pattern[m].append(pattern[pi][1])
    #                 continue

    #         if found_solution: 
    #             ub, lb = temp # feasible solution이면 p 길이가 3 고정
    #             temp_pattern[m].append(ub) 
    #             temp_pattern[m].append(lb) # pattern의 두번째 정보
    #         elif found_unknown: # time over
    #             schedule[m] = []
    #             temp_pattern[m].append(None)
    #         else: # infeasible 
    #             temp_pattern[m].append(False)

    #     return temp_pattern, pattern_information, schedule
    #     
    # def unified_gurobi(self, Gamma, Lambda, time_limit):
    #     # Parameters
    #     job_num = self.env.job_num
    #     machine_num = self.env.machine_num
    #     duration = self.env.duration
    #     deadline = self.env.deadline
    #     T = sum(duration[j] for j in range(job_num)) + job_num - 1 - min(duration)
    #     SPEC_CDF = self.env.spec_cdf
    #     SCALED_V = self.env.scaled_v

    #     # Initialize Gurobi model
    #     model = Model("unified_mip")
    #     model.setParam("TimeLimit", time_limit)

    #     # Variables
    #     relation = {}
    #     setup = {}
    #     start_time = {}
    #     allocation = {}

    #     for i in range(job_num):
    #         start_time[i] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=T, name=f'st_{i}')
    #         for j in range(job_num):
    #             relation[i, j] = model.addVar(vtype=GRB.BINARY, name=f'rel_{i}_{j}')
    #             setup[i, j] = model.addVar(vtype=GRB.BINARY, name=f'setup_{i}_{j}')
    #         for m in range(machine_num):
    #             allocation[i, m] = model.addVar(vtype=GRB.BINARY, name=f'alloc_{i}_{m}')

    #     for i in range(job_num):
    #         relation[i, -1] = model.addVar(vtype=GRB.BINARY, name=f'rel_{i}_{-1}')
    #         relation[-1, i] = model.addVar(vtype=GRB.BINARY, name=f'rel_{-1}_{i}')  

    #     # Objective: Minimize setup costs
    #     model.setObjective(quicksum(setup[i, j] for i in range(job_num) for j in range(job_num)), GRB.MINIMIZE)

    #     # Deadline constraints
    #     for j in range(job_num):
    #         assert duration[j] <= deadline[j]
    #         model.addConstr(start_time[j] + duration[j] <= deadline[j], f'deadline_{j}') # 5

    #     # Allocation constraints
    #     for i in range(job_num):
    #         model.addConstr(quicksum(allocation[i, m] for m in range(machine_num)) == 1, f'allocation_{i}') # 1

    #     model.addConstr(quicksum(relation[-1, i] for i in range(job_num)) == machine_num, "total_est") # 17
    #     model.addConstr(quicksum(relation[i, -1] for i in range(job_num)) == machine_num, "total_lst") # 18

    #     # Relation constraints
    #     for i in range(job_num):
    #         model.addConstr(quicksum(relation[i, j] for j in range(-1, job_num)) == 1, f'relation_row_{i}') # 10, 11
    #     for j in range(job_num):
    #         model.addConstr(quicksum(relation[i, j] for i in range(-1, job_num)) == 1, f'relation_col_{j}') # 7, 8
    #     for j in range(job_num):
    #         model.addConstr(relation[j, j] == 0, f'relation_diag_{j}') # 12
    #         for i in range(job_num):
    #             if j < i:
    #                 model.addConstr(relation[i, j] + relation[j, i] <= 1, f'relation_symmetry_{i}_{j}') # 13

    #     # Setup definition
    #     condition = model.addVars(job_num, job_num, machine_num, vtype=GRB.BINARY, name="condition")
    #     for i in range(job_num):
    #         for j in range(job_num):
    #             if self.job_to_family[i] != self.job_to_family[j]:
    #                 model.addConstr(setup[i, j] >= relation[i, j], f'setup_{i}_{j}') # 15
    #             else:
    #                 model.addConstr(setup[i, j] == 0, f'setup_zero_{i}_{j}') # 16
    #             if i != j:  
    #                 for m in range(machine_num):
    #                     model.addConstr(condition[i, j, m] <= (allocation[i, m] + allocation[j, m]))
    #                     model.addConstr(condition[i, j, m] <= 1 - (allocation[i, m] + allocation[j, m] - 1))
    #                     model.addConstr(condition[i, j, m] >= (allocation[i, m] - allocation[j, m]))
    #                     model.addConstr(condition[i, j, m] >= (allocation[j, m] - allocation[i, m]))

    #                     model.addConstr(relation[j, i] <= (1 - condition[i, j, m]), f'relation_{i}_{j}_{m}') # 4

    #     # Predecessor-successor constraints
    #     for i in range(job_num):
    #         for j in range(job_num):
    #             model.addConstr(
    #                 start_time[i] + duration[i] + setup[i, j] <= start_time[j] + T * (1 - relation[i, j]),
    #                 f'pred_succ_{i}_{j}'
    #             ) # 14

    #     # Constraint: Expected yield
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocation[j, m] * SPEC_CDF[j][m] for m in range(machine_num))
    #             for j in range(job_num)
    #         ) >= sum(duration) * Gamma,
    #         name="expected_yield"
    #     ) # 2

    #     # Constraint: Expected variance
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocation[j, m] * SCALED_V[j][m] for m in range(machine_num))
    #             for j in range(job_num)
    #         ) <= sum(duration) * Lambda,
    #         name="expected_variance"
    #     ) # 3

    #     # Solve the model
    #     model.optimize()

    #     # Check the solution status
    #     if model.status == GRB.OPTIMAL:
    #         print(int(model.ObjVal), "Optimal", round(model.Runtime, 2), "seconds")
    #         return True
    #     elif model.status == GRB.INFEASIBLE:
    #         return False
    #     else:
    #         print("Time limit exceeded")
    #         return None
    
    # def improved_cp_sequence_pattern(self, time_limit, pattern_information=None, pattern=None):
    #     if pattern_information is not None: print(f"pattern selection: {pattern_information}")
        
    #     # EDD
    #     self.EDD = self.use_EDD()
    #     for m in range(self.env.machine_num):
    #         self.count_dict[m] = Counter(tuple_item[-1] for tuple_item in self.EDD[m])
    #         self.machine_family_num[m] = len(self.count_dict[m].keys())
    #         print(f"Assigned jobs: {self.machine_jobs[m]}")
    #     print(f"Family number per machines: {self.machine_family_num}")

            
    #     temp_pattern = [[] for _ in range(self.env.machine_num)]
    #     for m in range(self.env.machine_num):
    #         temp_pattern[m].append([1 if j in self.machine_jobs[m] else 0 for j in range(self.env.job_num)])
    #         if pattern_information is not None:
    #             if pattern_information[m] is False:
    #                 temp = self.improved_cp_sequence_one_machine(m, time_limit)
    #             else:
    #                 index = pattern_information[m][0]
    #                 if len(pattern_information[m]) == 2:
    #                     ub_lb = (pattern_information[m][1][0], pattern_information[m][1][1])
    #                     temp = self.improved_cp_sequence_one_machine(m, time_limit, ub_lb)
    #                 else:
    #                     temp = (pattern[index][1], pattern[index][2])
    #                     print("status: OPTIMAL objective value:", temp[0], "lower bound:", temp[1])
    #         else:
    #             temp = self.improved_cp_sequence_one_machine(m, time_limit)

    #         if temp is not None: # feasible조차 안 나온 경우 예외 처리 필요하지만 EDD로 해결 가능
    #             ub, lb = temp # feasible solution이면 p 길이가 3 고정
    #             temp_pattern[m].append(ub) 
    #             temp_pattern[m].append(lb)
    #         else: # feasible조차 안 나온 경우
    #             temp_pattern[m].append(None)

    #     return temp_pattern

    # def cp_sequence(self, time_limit, hint:bool=False):
    #     # EDD
    #     if hint:
    #         self.EDD = self.use_EDD()
    #     answer = [0 for m in range(self.env.machine_num)]
    #     for m in range(self.env.machine_num):
    #         temp = self.cp_sequence_one_machine(m, time_limit, hint)
    #         if temp is not None:
    #             answer[m] = temp
    #     return answer

    # def cp_sequence_one_machine(self, machine_index, time_limit, hint:bool=False):
    #     # parameters
    #     job_list = self.machine_jobs[machine_index]
    #     if len(job_list) == 0:
    #         return None
    #     duration = self.env.duration
    #     deadline = self.env.deadline
    #     T = sum(duration[j] for j in job_list) + len(job_list)

    #     model = cp_model.CpModel()

    #     # Variables
    #     relation = {}
    #     setup = {}
    #     start_time = {}
    #     est = {}
    #     lst = {}

    #     for i in job_list:
    #         start_time[i] = model.NewIntVar(0, T, f'st_{i}')
    #         est[i] = model.NewBoolVar(f'est_{i}')
    #         lst[i] = model.NewBoolVar(f'lst_{i}')
    #         for j in job_list:
    #             relation[i, j] = model.NewBoolVar(f'rel_{i}_{j}')
    #             setup[i, j] = model.NewBoolVar(f'setup_{i}_{j}')    

    #     # objective
    #     model.minimize(sum([setup[i, j] for i in job_list for j in job_list]))
        
    #     # deadline
    #     for j in job_list:
    #         assert duration[j] <= deadline[j]
    #         model.add(start_time[j] + duration[j] <= deadline[j])

    #     # Earliest start time (est constraints)
    #     for i in job_list:
    #         for j in job_list:
    #             if i != j:
    #                 model.add(start_time[i] + duration[i] <= start_time[j]).only_enforce_if(est[i])
    #     # Ensure exactly one earliest start
    #     model.add(sum(est[i] for i in job_list) == 1)

    #     # Latest start time (lst constraints)
    #     for i in job_list:
    #         for j in job_list:
    #             if i != j:
    #                 model.add(start_time[j] + duration[j] <= start_time[i]).only_enforce_if(lst[i])
    #     # Ensure exactly one latest start
    #     model.add(sum(lst[i] for i in job_list) == 1)

    #     # 각 행의 합이 1이 되도록 제약 조건 추가
    #     for i in job_list:
    #         model.add(sum(relation[i, j] for j in job_list) == 1).only_enforce_if(~lst[i])
    #         model.add(sum(relation[i, j] for j in job_list) == 0).only_enforce_if(lst[i])

    #     # 각 열의 합이 1이 되도록 제약 조건 추가
    #     for j in job_list:
    #         model.add(sum(relation[i, j] for i in job_list) == 1).only_enforce_if(~est[j])
    #         model.add(sum(relation[i, j] for i in job_list) == 0).only_enforce_if(est[j])

    #     # 대각 예외 제약 조건 추가
    #     for j in job_list:
    #         model.add(relation[j, j] == 0)

    #     # setup 정의
    #     for i in job_list:
    #         for j in job_list:
    #             model.add(relation[i, j] + relation[j, i] <= 1)
    #             if self.job_to_family[i] != self.job_to_family[j]:
    #                 model.add(setup[i, j] >= relation[i, j])
    #             else:
    #                 model.add(setup[i, j] == 0)

    #     # predecessor-successor
    #     for i in job_list:
    #         for j in job_list:
    #             model.add(start_time[i] + duration[i] + setup[i, j] <= start_time[j]).only_enforce_if(relation[i, j])

    #     # EDD를 힌트로 사용, 더 많은 변수에 값 추가 가능하지만 코딩 필요
    #     if hint and getattr(self, "EDD"):
    #         for job, eddst, _, _ in self.EDD[machine_index]:
    #             model.add_hint(start_time[job], eddst)

    #     solver = cp_model.CpSolver()
    #     solver.parameters.max_time_in_seconds = time_limit
    #     solution_printer = SolutionPrinter(self.viz)
    #     status = solver.solve(model, solution_printer) # solver가 해결하도록
    #     print("status:", solver.status_name(status), "objective value:", solution_printer.obj_values[-1],  
    #           "lower bound:", solution_printer.lbd_values[-1], "time:", round(solution_printer.WallTime(), 2))
    #     if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    #         start_times = [(j, int(solver.Value(start_time[j])), int(duration[j]), self.job_to_family[j]) for j in job_list]
    #         self.schedule[machine_index] = sorted(start_times, key=lambda x: x[1])
    #         return int(solver.objective_value)
    #     else:
    #         return None

    # def improved_cp_sequence(self, time_limit, hint:bool=False):
    #     # EDD
    #     if hint:
    #         self.EDD = self.use_EDD()

    #     answer = [0 for m in range(self.env.machine_num)]
    #     for m in range(self.env.machine_num):
    #         temp = self.improved_cp_sequence_one_machine(m, time_limit, hint)
    #         if temp is not None:
    #             answer[m] = temp
    #     return answer
        
    # def gurobi_sequence(self, time_limit):
    #     answer = 0
    #     for m in range(self.env.machine_num):
    #         temp = self.gurobi_sequence_one_machine(m, time_limit)
    #         if temp is not None:
    #             answer += temp
    #     return answer

    # def gurobi_sequence_one_machine(self, machine_index, time_limit):
    #     # parameters
    #     job_list = self.machine_jobs[machine_index]
    #     if len(job_list) == 0:
    #         return None
    #     duration = self.env.duration
    #     deadline = self.env.deadline
    #     T = sum(duration[j] for j in job_list) + len(job_list)

    #     # Create Gurobi model
    #     model = Model()
    #     model.setParam('OutputFlag', self.viz)
    #     model.setParam('TimeLimit', time_limit)

    #     # Variables
    #     relation = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"rel_{i}_{j}") for i in job_list for j in job_list}
    #     setup = {(i, j): model.addVar(vtype=GRB.BINARY, name=f"setup_{i}_{j}") for i in job_list for j in job_list}
    #     start_time = {j: model.addVar(lb=0, ub=T, vtype=GRB.INTEGER, name=f"st_{j}") for j in job_list}
    #     est = {j: model.addVar(vtype=GRB.BINARY, name=f"est_{j}") for j in job_list}
    #     lst = {j: model.addVar(vtype=GRB.BINARY, name=f"lst_{j}") for j in job_list}

    #     # Objective: minimize total setup costs
    #     model.setObjective(quicksum(setup[i, j] for i in job_list for j in job_list), GRB.MINIMIZE)

    #     # Deadline constraints
    #     for j in job_list:
    #         assert duration[j] <= deadline[j], f"Job {j} has duration exceeding its deadline"
    #         model.addConstr(start_time[j] + duration[j] <= deadline[j], f"deadline_{j}")

    #     # earliest start time
    #     for i in job_list:
    #         for j in job_list:
    #             if i != j:
    #                 model.addGenConstrIndicator(
    #                             est[i],  # Indicator variable
    #                             True,              
    #                             start_time[i] + duration[i] <= start_time[j],  # Linear constraint
    #                             name=f"est_const_{i}_{j}"
    #                         )
    #     model.addConstr(quicksum(est[i] for i in job_list) == 1, f"est_sum_{i}")

    #     # latest start time
    #     for i in job_list:
    #         for j in job_list:
    #             if i != j:
    #                 model.addGenConstrIndicator(
    #                             lst[i],  # Indicator variable
    #                             True,              
    #                             start_time[j] + duration[j] <= start_time[i],  # Linear constraint
    #                             name=f"lst_const_{j}_{i}"
    #                         )
    #     model.addConstr(quicksum(lst[i] for i in job_list) == 1, f"lst_sum_{i}")

    #     # Row and column sum constraints for the relation matrix
    #     for i in job_list: # successor may not be exist
    #         model.addGenConstrIndicator(
    #                         lst[i],  # Indicator variable
    #                         False,              
    #                         quicksum(relation[i, j] for j in job_list) == 1,  # Linear constraint
    #                         name=f"row_sum_{i}"
    #                     )
    #         model.addGenConstrIndicator(
    #                         lst[i],  # Indicator variable
    #                         True,              
    #                         quicksum(relation[i, j] for j in job_list) == 0,  # Linear constraint
    #                         name=f"est_row_sum_{i}"
    #                     )
            
    #     for j in job_list: # precedence may not be exist
    #         model.addGenConstrIndicator(
    #                         est[j],  # Indicator variable
    #                         False,              
    #                         quicksum(relation[i, j] for i in job_list) == 1,  # Linear constraint
    #                         name=f"col_sum_{j}"
    #                     )
    #         model.addGenConstrIndicator(
    #                         est[j],  # Indicator variable
    #                         True,              
    #                         quicksum(relation[i, j] for i in job_list) == 0,  # Linear constraint
    #                         name=f"lst_col_sum_{j}"
    #                     )

    #     # Diagonal exception (no self-loop)
    #     for j in job_list:
    #         model.addConstr(relation[j, j] == 0, f"diag_{j}")

    #     # Setup constraints
    #     for i in job_list:
    #         for j in job_list:
    #             model.addConstr(relation[i, j] + relation[j, i] <= 1, f"relation_sym_{i}_{j}")
    #             if self.job_to_family[i] != self.job_to_family[j]:
    #                 model.addConstr(setup[i, j] >= relation[i, j], f"setup_diff_{i}_{j}")
    #             else:
    #                 model.addConstr(setup[i, j] == 0, f"setup_same_{i}_{j}")

    #     # Predecessor-successor constraints
    #     for i in job_list:
    #         for j in job_list:
    #             model.addGenConstrIndicator(
    #                 relation[i, j],  # Indicator variable
    #                 True,              # Active when relation[i, j] == 1
    #                 start_time[i] + duration[i] + setup[i, j] <= start_time[j],  # Linear constraint
    #                 name=f"precedence_{i}_{j}"
    #             )

    #     # Solve the model
    #     model.optimize()
    #     if model.Status == GRB.INFEASIBLE or model.Status == GRB.UNBOUNDED:
    #         print("Model is infeasible")
    #         # model.computeIIS()
    #         # model.write("infeasible_model.ilp")  # Exports the IIS to a file for analysis
    #         return None

    #     if model.status == GRB.OPTIMAL:
    #         print("Model is optimal")
    #         start_times = [(j, int(start_time[j].X), int(duration[j]), self.job_to_family[j]) for j in job_list]
    #         self.schedule[machine_index] = sorted(start_times, key=lambda x: x[1])
    #         return int(model.objVal)
    #     print("Time Limit over")
    #     return None
        
    # def gurobi_allocation_makespan(self, Gamma, Lambda, time_limit):
    #     # parameters
    #     job_num = self.env.job_num
    #     machine_num = self.env.machine_num
    #     family_num = self.env.family_num
    #     family = self.env.family
    #     duration = self.env.duration
    #     spec_cdf = self.env.spec_cdf
    #     scaled_v = self.env.scaled_v

    #     model = Model("Allocation")
    #     model.setParam('OutputFlag', self.viz)
    #     model.setParam("TimeLimit", time_limit)  # 시간 제한 설정

    #     # Variables
    #     allocations = model.addVars(job_num, machine_num, vtype=GRB.BINARY, name="allocations")
    #     c_max = model.addVar(lb=0, ub=np.sort(duration)[-job_num:].sum(), vtype=GRB.CONTINUOUS, name="c_max")

    #     # Objective: Minimize c_max
    #     model.setObjective(c_max, GRB.MINIMIZE)

    #     # Constraint: c_max
    #     for m in range(machine_num):
    #         model.addConstr(
    #             quicksum(allocations[j, m] * duration[j] for j in job_list) <= c_max,
    #             name=f"c_max_machine_{m}"
    #         )

    #     # Constraint: Job uniqueness
    #     for j in job_list:
    #         model.addConstr(
    #             quicksum(allocations[j, m] for m in range(machine_num)) == 1,
    #             name=f"job_uniqueness_{j}"
    #         )

    #     # Optional Constraint: Ban list
    #     for m in range(machine_num):
    #         for i, j in self.ban_list[m]:
    #             model.addConstr(allocations[i, m] + allocations[j, m] <= 1,
    #                 name=f"ban_list_{j}"
    #             )

    #     # Constraint: Expected yield
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * spec_cdf[j][m] for m in range(machine_num))
    #             for j in job_list
    #         ) >= sum(duration) * Gamma,
    #         name="expected_yield"
    #     )

    #     # Constraint: Expected variance
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * scaled_v[j][m] for m in range(machine_num))
    #             for j in job_list
    #         ) <= sum(duration) * Lambda,
    #         name="expected_variance"
    #     )

    #     # Solve the model
    #     model.optimize()

    #     # Check the solution
    #     self.time = model.Runtime
    #     self.status = model.Status
    #     if model.Status == GRB.INFEASIBLE or model.Status == GRB.UNBOUNDED:
    #         print("No feasible solution found.", round(model.Runtime, 2), "seconds")
    #         return None
        
    #     self.allocation = np.zeros((job_num, machine_num), dtype=bool)
    #         # Fill the numpy array with variable values
    #     for j in job_list:
    #         for m in range(machine_num):
    #             self.allocation[j, m] = bool(allocations[j, m].X)  # Extract the value of allocations[j, m]
                
    #     if model.Status == GRB.OPTIMAL:
    #         print(int(model.ObjVal), "Optimal", round(model.Runtime, 2), "seconds")
    #         return int(model.ObjVal)
    #     print(int(model.ObjVal), "Feasible", round(model.Runtime, 2), "seconds")
    #     return int(model.ObjVal)
    
    # def gurobi_allocation_max_setup(self, Gamma, Lambda, time_limit, bigm=False):
    #     # parameters
    #     job_num = self.env.job_num
    #     machine_num = self.env.machine_num
    #     family_num = self.env.family_num
    #     family = self.env.family
    #     duration = self.env.duration
    #     spec_cdf = self.env.spec_cdf
    #     scaled_v = self.env.scaled_v

    #     model = Model("Allocation")
    #     model.setParam('OutputFlag', self.viz)
    #     model.setParam("TimeLimit", time_limit)  # 시간 제한 설정

    #     # Variables
    #     allocations = model.addVars(job_num, machine_num, vtype=GRB.BINARY, name="allocations")
    #     s_max = model.addVar(lb=0, ub=job_num, vtype=GRB.CONTINUOUS, name="c_max")

    #     machine_max_family = model.addVars(machine_num, vtype=GRB.CONTINUOUS, name="machine_max_family")
    #     machine_sub_max_family = model.addVars(machine_num, vtype=GRB.CONTINUOUS, name="machine_sub_max_family")
    #     machine_check_max = model.addVars(machine_num, family_num, vtype=GRB.BINARY, name="machine_check_max")

    #     # Objective: Minimize c_max
    #     model.setObjective(s_max, GRB.MINIMIZE)

    #     # Constraint: s_max
    #     for m in range(machine_num):
    #         model.addConstr(
    #             quicksum(quicksum(allocations[j, m] for j in family[f]) for f in range(family_num)) 
    #                 - machine_max_family[m] + machine_sub_max_family[m] <= s_max,
    #             name=f"s_max_machine_{m}"
    #         )

    #     # Constraint: machine_family_relation
    #     for m in range(machine_num):
    #         model.addConstr(quicksum(machine_check_max[m, f] for f in range(family_num)) == 1, name=f"machine_check_max_{m}")
                
    #     for m in range(machine_num):
    #         model.addConstrs((machine_max_family[m] >= quicksum(allocations[f, m] for f in family[i]) for i in range(family_num)), 
    #                          name=f"machine_max_family_lb_{m}")
    #         model.addConstr(machine_sub_max_family[m] <= machine_max_family[m], name=f"machine_sub_max_family_ub_{m}")
    #         if bigm:
    #             M = 1e3
    #             model.addConstrs((machine_max_family[m] <= quicksum(allocations[f, m] for f in family[i]) + M * (1 - machine_check_max[m, i]) for i in range(family_num)), 
    #                             name=f"machine_max_family_ub_{m}")
    #             model.addConstrs((machine_sub_max_family[m] >= quicksum(allocations[f, m] for f in family[i]) - M * machine_check_max[m, i] for i in range(family_num)), 
    #                             name=f"machine_sub_max_family_lb_{m}")
    #         else:
    #             for i in range(family_num):
    #                 model.addGenConstrIndicator(machine_check_max[m, i], True, machine_max_family[m] <= quicksum(allocations[f, m] for f in family[i]), name=f"machine_max_family_ub_{m}{i}")
    #                 model.addGenConstrIndicator(machine_check_max[m, i], False, machine_sub_max_family[m] >= quicksum(allocations[f, m] for f in family[i]), name=f"machine_sub_max_family_lb_{m}{i}")
                
            
    #     # Constraint: Job uniqueness
    #     for j in job_list:
    #         model.addConstr(
    #             quicksum(allocations[j, m] for m in range(machine_num)) == 1,
    #             name=f"job_uniqueness_{j}"
    #         )

    #     # Optional Constraint: Ban list
    #     for m in range(machine_num):
    #         for i, j in self.ban_list[m]:
    #             model.addConstr(allocations[i, m] + allocations[j, m] <= 1,
    #                 name=f"ban_list_{j}"
    #             )

    #     # Constraint: Expected yield
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * spec_cdf[j][m] for m in range(machine_num))
    #             for j in job_list
    #         ) >= sum(duration) * Gamma,
    #         name="expected_yield"
    #     )

    #     # Constraint: Expected variance
    #     model.addConstr(
    #         quicksum(
    #             duration[j] * quicksum(allocations[j, m] * scaled_v[j][m] for m in range(machine_num))
    #             for j in job_list
    #         ) <= sum(duration) * Lambda,
    #         name="expected_variance"
    #     )

    #     # Solve the model
    #     model.optimize()

    #     # Check the solution
    #     self.time = model.Runtime
    #     self.status = model.Status
    #     if model.Status == GRB.INFEASIBLE or model.Status == GRB.UNBOUNDED:
    #         print("No feasible solution found.", round(model.Runtime, 2), "seconds")
    #         return None
        
    #     self.allocation = np.zeros((job_num, machine_num), dtype=bool)
    #         # Fill the numpy array with variable values
    #     for j in job_list:
    #         for m in range(machine_num):
    #             self.allocation[j, m] = bool(allocations[j, m].X)  # Extract the value of allocations[j, m]
                
    #     if model.Status == GRB.OPTIMAL:
    #         print(int(model.ObjVal), "Optimal", round(model.Runtime, 2), "seconds")
    #         return int(model.ObjVal)
    #     print(int(model.ObjVal), "Feasible", round(model.Runtime, 2), "seconds")
    #     return int(model.ObjVal)
    
    # def gurobi_allocation_min_setup(self, Gamma, Lambda, time_limit):
        # parameters
        # job_num = self.env.job_num
        # machine_num = self.env.machine_num
        # family_num = self.env.family_num
        # family = self.env.family
        # duration = self.env.duration
        # spec_cdf = self.env.spec_cdf
        # scaled_v = self.env.scaled_v

        # model = Model("Allocation")
        # model.setParam('OutputFlag', self.viz)
        # model.setParam("TimeLimit", time_limit)  # 시간 제한 설정

        # # Variables
        # allocations = model.addVars(job_num, machine_num, vtype=GRB.BINARY, name="allocations")
        # s_min = model.addVar(lb=0, ub=job_num, vtype=GRB.CONTINUOUS, name="c_max")

        # machine_family = model.addVars(machine_num, family_num, vtype=GRB.BINARY, name="machine_check_max")

        # # Objective: Minimize c_max
        # model.setObjective(s_min, GRB.MINIMIZE)

        # # Constraint: s_max
        # for m in range(machine_num):
        #     model.addConstr(
        #         quicksum(machine_family[m, f] for f in range(family_num)) - 1 <= s_min,
        #         name=f"s_min_machine_{m}"
        #     )

        # # Constraint: machine_family_relation
        # for m in range(machine_num):
        #     for i in range(family_num):
        #         model.addGenConstrIndicator(machine_family[m, i], True, quicksum(allocations[f, m] for f in family[i]) >= 1, name=f"machine_family_exist_{m}{i}")
        #         model.addGenConstrIndicator(machine_family[m, i], False, quicksum(allocations[f, m] for f in family[i]) <= 0, name=f"machine_family_not_exist_{m}{i}")

        # # Constraint: Job uniqueness
        # for j in job_list:
        #     model.addConstr(
        #         quicksum(allocations[j, m] for m in range(machine_num)) == 1,
        #         name=f"job_uniqueness_{j}"
        #     )

        # # Optional Constraint: Ban list
        # for m in range(machine_num):
        #     for i, j in self.ban_list[m]:
        #         model.addConstr(allocations[i, m] + allocations[j, m] <= 1,
        #             name=f"ban_list_{j}"
        #         )

        # # Constraint: Expected yield
        # model.addConstr(
        #     quicksum(
        #         duration[j] * quicksum(allocations[j, m] * spec_cdf[j][m] for m in range(machine_num))
        #         for j in job_list
        #     ) >= sum(duration) * Gamma,
        #     name="expected_yield"
        # )

        # # Constraint: Expected variance
        # model.addConstr(
        #     quicksum(
        #         duration[j] * quicksum(allocations[j, m] * scaled_v[j][m] for m in range(machine_num))
        #         for j in job_list
        #     ) <= sum(duration) * Lambda,
        #     name="expected_variance"
        # )

        # # Solve the model
        # model.optimize()

        # # Check the solution
        # self.time = model.Runtime
        # self.status = model.Status
        # if model.Status == GRB.INFEASIBLE or model.Status == GRB.UNBOUNDED:
        #     print("No feasible solution found.", round(model.Runtime, 2), "seconds")
        #     return None
        
        # self.allocation = np.zeros((job_num, machine_num), dtype=bool)
        #     # Fill the numpy array with variable values
        # for j in job_list:
        #     for m in range(machine_num):
        #         self.allocation[j, m] = bool(allocations[j, m].X)  # Extract the value of allocations[j, m]
                
        # if model.Status == GRB.OPTIMAL:
        #     print(int(model.ObjVal), "Optimal", round(model.Runtime, 2), "seconds")
        #     return int(model.ObjVal)
        # print(int(model.ObjVal), "Feasible", round(model.Runtime, 2), "seconds")
        # return int(model.ObjVal)

    # def improved_cp_sequence_due_feasible(self, time_limit):
    #     # parameters
    #     job_num = self.env.job_num
    #     machine_num = self.env.machine_num
    #     duration = self.env.duration
    #     deadline = self.env.deadline
    #     T = sum(duration[j] for j in range(job_num)) - min(duration)

    #     model = cp_model.CpModel()

    #     # Variables
    #     start_time = {}
    #     allocation = {}
    #     relation = {}
    #     for i in range(job_num):
    #         start_time[i] = model.NewIntVar(0, T, f'st_{i}')
    #         for j in range(job_num):
    #             relation[i, j] = model.NewBoolVar(f'rel_{i}_{j}')
    #         for m in range(machine_num):
    #             allocation[i, m] = model.NewBoolVar(f'alloc_{i}_{m}')

    #     # deadline
    #     for j in range(job_num):
    #         assert duration[j] <= deadline[j]
    #         model.add(start_time[j] + duration[j] <= deadline[j])

    #     # allocation
    #     for i in range(job_num):
    #         model.add(sum(allocation[i, m] for m in range(machine_num)) == 1)

    #     # 할당 - 같은 머신 변수 관계
    #     for i in range(job_num):
    #         for j in range(job_num):
    #             if i < j:
    #                 for m in range(machine_num):
    #                     model.add(relation[i, j] == 1).only_enforce_if(allocation[i, m], allocation[j, m])
                        
    #                     condition = model.NewBoolVar(f"temp_{i}_{j}_{m}")
    #                     model.AddBoolOr([~allocation[i, m], ~allocation[j, m]]).OnlyEnforceIf(condition)
    #                     model.add(relation[i, j] == 0).only_enforce_if(condition)
                        
    #     # 같은 머신 변수 대칭성
    #     for i in range(job_num):  
    #         for j in range(job_num):
    #             if i != j:  
    #                 model.add(relation[i, j] == relation[j, i])
    #             else:
    #                 model.add(relation[i, j] == 1)

    #     # predecessor-successor
    #     for i in range(job_num):  
    #         for j in range(job_num):
    #             if i != j:
    #                 for m in range(machine_num):
    #                     no_overlap = model.NewBoolVar(f'no_overlap_{i}_{j}_{m}')
    #                     model.add(start_time[i] + duration[i] <= start_time[j]).only_enforce_if(no_overlap).only_enforce_if(relation[i, j])
    #                     model.add(start_time[j] + duration[j] <= start_time[i]).only_enforce_if(no_overlap.Not()).only_enforce_if(relation[i, j])

    #     solver = cp_model.CpSolver()
    #     # solution_printer = SolutionPrinter(self.viz)
    #     solver.parameters.max_time_in_seconds = time_limit
    #     status = solver.solve(model) # solver가 해결하도록

    #     print("status:", solver.status_name(status), "time:", round(solver.WallTime(), 2))
    #     if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    #         return True
    #     elif status == cp_model.INFEASIBLE:
    #         return False
    #     else:
    #         print("time limit over")
    #         return None