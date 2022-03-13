# 所有排序都实现的是从小至大的排序


# 1冒泡排序（优化：就是立一个flag，当在一趟序列遍历中元素没有发生交换，则证明该序列已经有序。但这种改进对于提升性能来说并没有什么太大作用。）
# 平均时间复杂度O(n**2)，空间复杂度O(1)，稳定
def bubbleSort(arr):
    for i in range(1, len(arr)):  # 每一个i循环里，都将序列的最后一位确定下来
        for j in range(0, len(arr) - i):  # 实现两个相邻的‘大、小’数字交换
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


# 2选择排序（在未排序队列中，每次都找最小的元素放在最开始的位置。希望数据越少越好）
# 平均时间复杂度O(n**2)，空间复杂度O(1)，不稳定
def selectSort(arr):
    for i in range(len(arr) - 1):  # 每次都挑出最小的数放在最前面
        min_index = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_index]:  # 找到最小数，记录位置
                min_index = j
        if min_index != i:
            arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr


# 3插入排序（思想类似于扑克牌捋牌。通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。）
# 平均时间复杂度O(n**2)，空间复杂度O(1)，稳定
def insertionSort(arr):
    for i in range(len(arr)):  # 左边开始到preIndex是已排序好的序列，想要将current插入到里面
        preIndex = i - 1
        current = arr[i]
        while preIndex >= 0 and arr[preIndex] > current:  # 序列存在 且最大数大于现在还没排序的数current
            arr[preIndex + 1] = arr[preIndex]  # 从最后一位开始，向右移动一位，直到这一位小于current
            preIndex -= 1
        arr[preIndex + 1] = current  # 将current插入到序列里
    return arr


# 4希尔排序（插入排序的改进：先将整个序列分割成为若干子序列分别进行直接插入排序，待整个序列中“基本有序”时，再对全体记录进行依次直接插入排序。）
# 平均时间复杂度O(n*logn)，空间复杂度O(1)，不稳定
def shellSort(arr):
    import math
    gap=1
    while(gap < len(arr)/3):  # 也可以除其他数字，不一定是3，只是将他们分组
        gap = gap*3+1
    while gap > 0:
        for i in range(gap,len(arr)):
            temp = arr[i]
            j = i-gap
            while j >=0 and arr[j] > temp:
                arr[j+gap]=arr[j]
                j-=gap
            arr[j+gap] = temp
        gap = math.floor(gap/3)
    return arr


# 5归并排序(分治法,归并操作：可以自上而下的递归，也可以自下而上的迭代)
# 平均时间复杂度O(n*logn)，空间复杂度O(n)，稳定
def mergeSort(arr):
    import math
    if(len(arr)<2):
        return arr
    middle = math.floor(len(arr)/2)
    left, right = arr[0:middle], arr[middle:]
    return merge(mergeSort(left), mergeSort(right))

def merge(left,right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0));
        else:
            result.append(right.pop(0));
    while left:
        result.append(left.pop(0));
    while right:
        result.append(right.pop(0));
    return result


# 6快速排序（明显比其他 Ο(nlogn) 算法更快，算是在冒泡排序基础上的递归分治法。每次从数列中挑出一个“基准”，比它小的放前面，大的放后面）
# 平均时间复杂度O(n*logn)，空间复杂度O(n*logn)，不稳定
def quickSort(arr, left=None, right=None):
    left = 0 if not isinstance(left,(int, float)) else left
    right = len(arr)-1 if not isinstance(right,(int, float)) else right
    if left < right:
        partitionIndex = partition(arr, left, right)
        quickSort(arr, left, partitionIndex-1)
        quickSort(arr, partitionIndex+1, right)
    return arr

def partition(arr, left, right):
    pivot = left
    index = pivot+1
    i = index
    while  i <= right:
        if arr[i] < arr[pivot]:
            swap(arr, i, index)
            index+=1
        i+=1
    swap(arr,pivot,index-1)
    return index-1

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


# 7堆排序（与其他语言不同，python自带的heapq包中实现的是小顶堆）
# 平均时间复杂度O(n*logn)，空间复杂度O(1)，不稳定
def heapify(arr, n, i):  # 检查是否满足根大于孩子，不满足时交换根与孩子的位置
    largest = i
    l = 2 * i + 1  # left = 2*i + 1
    r = 2 * i + 2  # right = 2*i + 2
    if l < n and arr[i] < arr[l]:  # 调用前声明n=len(arr)
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # 交换
        heapify(arr, n, largest)


def heapSort(arr):  # 实现堆排序
    n = len(arr)
    # Build a maxheap.
    for i in range(n, -1, -1):  # 从最后一个结点开始
        heapify(arr, n, i)  # 让结点的孩子都比它小
    # 一个个交换元素
    for i in range(n - 1, 0, -1):  # （每次都找出最大值，放到最后，这样就实现了从小到大排列）
        arr[i], arr[0] = arr[0], arr[i]  # 交换根（堆中的最大值）和最末的结点
        heapify(arr, i, 0)  # 除去移动后的根结点，其余结点重新构造大顶堆


# 8计数排序：将输入的数据值转化为键存储在额外开辟的数组空间中。要求输入的数据必须是有确定范围的整数。
# 平均时间复杂度O(n+k)，空间复杂度O(n)，稳定
def countingSort(arr, maxValue):
    bucketLen = maxValue+1
    bucket = [0]*bucketLen
    sortedIndex =0
    arrLen = len(arr)
    for i in range(arrLen):
        if not bucket[arr[i]]:
            bucket[arr[i]]=0
        bucket[arr[i]]+=1
    for j in range(bucketLen):
        while bucket[j]>0:
            arr[sortedIndex] = j
            sortedIndex+=1
            bucket[j]-=1
    return arr


# 9桶排序（桶先将数据分到有限数量的桶里，然后对每一个桶内的数据进行排序(桶内排序可以使用任何一种排序算法)，最后将所有排好序的桶合并）
# 平均时间复杂度O(n+k)，空间复杂度O(n+k)，稳定
def bucket_sort(array):
    min_num, max_num = min(array), max(array)
    bucket_num = (max_num-min_num)//3 + 1
    buckets = [[] for _ in range(int(bucket_num))]
    for num in array:
        buckets[int((num-min_num)//3)].append(num)
    new_array = list()
    for i in buckets:
        for j in sorted(i):
            new_array.append(j)
