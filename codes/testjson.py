import json

name = 'Jan Dustin Tengdyantono'
phone = '2533322566'
email = 'dustinteng12@gmail.com'
left = 1
bot = 2
right = 3
top = 4
data = []
for i in range (5):
    data.append((i,left*i,bot*i,right*i,top*i)) #kiri bawah kanan atas
print(data)
# data_save = {
#     "name" : name
#     "phone" : phone
#     "email" : email
#     "data" : data
# }