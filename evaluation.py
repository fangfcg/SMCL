from experiment import *
import dataloader
import os

def eval(mr, dataset, k, lmd=10, iter=500, version=""):
    print(dataset)
    mat, missing_mat_list = dataloader.loadDataset(dataset, mr)

    methods = ['SMC-sgd', 'SMC-multi', 'SMCL_sgd', 'SMCL_multi']
    results = []
    times = []
    for i in range(len(missing_mat_list)):
        results.append([])
        times.append([])

    for idx, mat_missing in enumerate(missing_mat_list):

        SMC_sgd, _ , _, t1= SMCTest(mat, mat_missing, lmd, method=1, k=k, iter=iter)
        print('SMC-sgd: {}'.format(SMC_sgd))
        results[idx].append(SMC_sgd)
        times[idx].append(t1)

        SMC_multi, _ , _, t2= SMCTest(mat, mat_missing, lmd, method=2, k=k, iter=iter)
        print('SMC-multi: {}'.format(SMC_multi))
        results[idx].append(SMC_multi)
        times[idx].append(t2)


        SMCL_sgd, _, _, t5 = SMCLTest(mat, mat_missing, lmd, method=1, k=k, iter=iter, cluster=True)
        print('SMCL_sgd: {}'.format(SMCL_sgd))
        results[idx].append(SMCL_sgd)
        times[idx].append(t5)

        SMCL_multi, _, _, t6 = SMCLTest(mat, mat_missing, lmd, method=2, k=k, iter=iter, cluster=True)
        print('SMCL_multi: {}'.format(SMCL_multi))
        results[idx].append(SMCL_multi)
        times[idx].append(t6)



    results = np.array(results)
    times = np.array(times)
    print(results)
    result_avg = np.mean(results, axis=0)
    times_avg = np.mean(times, axis=0)
    print(result_avg)
    print(times_avg)
    print("average")
    newstr = ""
    for i in range(len(methods)):
        newstr += f'{methods[i]}:{result_avg[i]}'
        newstr += '\n'
        print(f'{methods[i]}:{result_avg[i]}')
    if not os.path.exists("results"):
        os.mkdir("results")
    with open(f"results/{version}-{dataset}-{mr}.txt", "w") as outputFile:
        outputFile.write(newstr)
        outputFile.close()

    newstr = ""
    for i in range(len(methods)):
        newstr += f'{methods[i]}:{times_avg[i]}'
        newstr += '\n'
    with open(f"results/{version}-{dataset}-{mr}-time.txt", "w") as outputFile:
        outputFile.write(newstr)
        outputFile.close()



if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    parameters = {
        'k': 6,
        'lmd': 10,
        'iter': 500,
    }
    version = "test"
    for mr in [50]:
        for dataset in ['kc', 'lake', 'farm', 'economic', 'california']:
            eval(mr, dataset, k=parameters['k'], lmd=parameters['lmd'], iter=parameters['iter'], version=version)

    print(f"version:{version} finished")