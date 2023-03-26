# 打开mdata.txt文件，读取所有行
file = open('mdata.txt', 'r')
lines = file.readlines()
# 关闭文件
file.close()
# 创建一个空的列表，用于存储每一行的数据和第一个数字
data = []
# 循环遍历每一行，用空格分割数据，将第一个数字转换为浮点数，添加到列表中
for line in lines:
  line = line.strip() # 去掉换行符
  nums = line.split(' ') # 用空格分割数据
  first = float(nums[0]) # 将第一个数字转换为浮点数
  data.append([first, line]) # 添加到列表中
# 对列表按照第一个数字的大小进行排序
data.sort()
# 打开mdata.txt文件，清空内容
file = open('mdata.txt', 'w')
# 循环遍历排序后的列表，将每一行的数据写入文件中
for item in data:
  file.write(item[1] + '\n')
# 关闭文件
file.close()