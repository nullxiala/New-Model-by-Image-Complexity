import math
def density(boxes, nums):
    boxes = boxes[0][:int(nums)]
    targets = []
    for box in boxes:
        x = (box[0] + box[2]) * 5
        y = (box[1] + box[3]) * 5
        targets.append((x,y))
    min_density = float('inf')
    min_point = None
    for target in targets:
        density = 0
        for point in targets:
            density += (point[0]-target[0])**2 + (point[1]-target[1])**2
        density /= len(targets)
        if density < min_density:
            min_density = density
            min_point = target
    count = 0
    for point in targets:
        distance = (point[0]-min_point[0])**2 + (point[1]-min_point[1])**2
        if distance < 2 * min_density:
            count += 1
    new_density = (10 - math.sqrt(min_density))**2
    return float(new_density) * count

def overlap_area(boxes, nums):
    boxes = boxes[0][:int(nums)]
    targets = []
    for box in boxes:
        x1 = box[0] * 10
        y1 = box[1] * 10
        x2 = box[2] * 10
        y2 = box[3] * 10
        targets.append((x1,y1,x2,y2))
    total_area = 0 # 初始化重合部分面积之和为0
    tot= 0
    # 遍历所有线框
    for i in range(len(targets)): # 遍历每个矩形框
        for j in range(i+1, len(targets)): # 遍历与之后的矩形框比较
            x1,y1,x2,y2 = targets[i] # 取出第i个矩形框的坐标
            x3,y3,x4,y4 = targets[j] # 取出第j个矩形框的坐标
            if x3 >= x2 or x4 <= x1 or y3 >= y2 or y4 <= y1: # 如果两个矩形框没有重合，就跳过
                continue
            else: # 否则，计算重合部分的面积，并累加到总面积中
                tot += abs((x2 - x1) * (y2 - y1))
                tot += abs((x4 - x3) * (y4 - y3))
                overlap_x1 = max(x1,x3)
                overlap_y1 = max(y1,y3)
                overlap_x2 = min(x2,x4)
                overlap_y2 = min(y2,y4)
                overlap_width = overlap_x2 - overlap_x1
                overlap_height = overlap_y2 - overlap_y1
                overlap_area = overlap_width * overlap_height
                total_area += overlap_area
    return float(total_area) # 返回总面积


