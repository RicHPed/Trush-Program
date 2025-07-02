from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# 算法数据结构
ALGORITHMS = {
    "排序算法": {
        "冒泡排序": {
            "description": "简单的排序算法，重复遍历要排序的数列",
            "complexity": "O(n²)",
            "code": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
""",
            "visualization": "bubble_sort"
        },
        "快速排序": {
            "description": "使用分治策略的高效排序算法",
            "complexity": "O(n log n)",
            "code": """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""",
            "visualization": "quick_sort"
        },
        "归并排序": {
            "description": "稳定的分治排序算法",
            "complexity": "O(n log n)",
            "code": """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
""",
            "visualization": "merge_sort"
        }
    },
    "搜索算法": {
        "二分搜索": {
            "description": "在有序数组中查找目标值的高效算法",
            "complexity": "O(log n)",
            "code": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
""",
            "visualization": "binary_search"
        },
        "线性搜索": {
            "description": "逐个检查数组元素的简单搜索算法",
            "complexity": "O(n)",
            "code": """
def linear_search(arr, target):
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1
""",
            "visualization": "linear_search"
        }
    },
    "图算法": {
        "深度优先搜索": {
            "description": "沿着图的深度遍历节点的算法",
            "complexity": "O(V + E)",
            "code": """
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start, end=' ')
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    
    return visited
""",
            "visualization": "dfs"
        },
        "广度优先搜索": {
            "description": "按层次遍历图的算法",
            "complexity": "O(V + E)",
            "code": """
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        vertex = queue.popleft()
        print(vertex, end=' ')
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
""",
            "visualization": "bfs"
        }
    },
    "动态规划": {
        "斐波那契数列": {
            "description": "经典的动态规划问题",
            "complexity": "O(n)",
            "code": """
def fibonacci_dp(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
""",
            "visualization": "fibonacci"
        },
        "最长公共子序列": {
            "description": "寻找两个序列的最长公共子序列",
            "complexity": "O(mn)",
            "code": """
def lcs(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
""",
            "visualization": "lcs"
        }
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/algorithms')
def get_algorithms():
    return jsonify(ALGORITHMS)

@app.route('/api/algorithm/<category>/<name>')
def get_algorithm(category, name):
    try:
        algorithm = ALGORITHMS[category][name]
        return jsonify({
            "category": category,
            "name": name,
            "data": algorithm
        })
    except KeyError:
        return jsonify({"error": "Algorithm not found"}), 404

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    # 模拟AI回复（实际应用中这里会调用真实的AI API）
    ai_response = f"我收到了您的消息：'{user_message}'。这是一个模拟的AI回复。在实际应用中，这里会调用真实的AI模型来生成代码和可视化。"
    
    return jsonify({
        "response": ai_response,
        "code": "# 这里会生成相关的代码",
        "visualization": "这里会生成可视化数据"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 