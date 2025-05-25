def all_permutations():
    def permute(nums, path, used, res):
        if len(path) == len(nums):
            res.append(tuple(path))
            return
        for i in range(len(nums)):
            if not used[i]:
                used[i] = True
                path.append(nums[i])
                permute(nums, path, used, res)
                path.pop()
                used[i] = False

    nums = [2, 3, 4, 5, 6, 7]
    res = []
    used = [False] * len(nums)
    permute(nums, [], used, res)
    return res

dic = {'Bruce': (5, 2, 3, 4, 6, 7), 
        'Tony': (2, 4, 3, 5, 6, 7), 
        'Jeffrey': (5, 2, 3, 6, 7, 4), 
        'Franklin': (4, 2, 3, 5, 6, 7), 
        'Jack': (4, 3, 5, 7, 6, 2), 
        'Terrence': (2, 3, 4, 7, 6, 5)}

list = [dic['Bruce'], dic['Tony'], dic['Jeffrey'], dic['Franklin'], dic['Jack'], dic['Terrence']]

def find_least_square(list, permutations):
    d2 = {}
    for i in permutations:
        score = 0
        for j in range(len(i)):
            score += list[j].index(i[j]) ** 2
        d2.setdefault(score, [])
        d2[score].append(i)
    if d2.keys():
        return d2[min(d2.keys())]
    return []

def find_least_variance(list, permutations):
    d2 = {}
    for p in permutations:
        mean = 0
        for i in range(len(p)):
            mean += list[i].index(p[i])
        mean /= len(p)
        variance = 0
        for i in range(len(p)):
            variance += (list[i].index(p[i]) - mean) ** 2
        variance /= len(p)
        d2.setdefault(variance, [])
        d2[variance].append(p)
    if d2.keys():
        print(min(d2.keys()))
        return d2[min(d2.keys())]
    return []
    

def find_all_less_than_2(list, permutations):
    l2 = []
    for p in permutations:
        count = 0
        for i in range(len(p)):
            if list[i].index(p[i]) > 2:
                count += 1
        if count < 3:
            l2.append(p)
    return l2

if __name__ == "__main__":
    for i in find_least_variance(list, all_permutations()):
        mean = 0
        for j in range(len(i)):
            mean += list[j].index(i[j])
        print('Bruce:', i[0],';', 'Tony:', i[1], ';', 'Jeffrey:', i[2], ';', 'Franklin:', i[3], ';', 'Jack:', i[4], ';', 'Terrence:', i[5], ';', 'mean:', mean / 6)

