#print(100*5+40*8)
#print(0.1+0.2)
#print(abs(-5))
#print(round(1.12, 1))
#import math
#print(math.sqrt(2))
#import calendar
#print(calendar.month(2024, 6))
#text = '123.4'
#print(int(float(text)))
#name = input("Enter your name: ")
#print(f"Hello, {name}!")
#text = input("アルファベットを入力してください:")
#lower_text = text.lower()
#print(text,'の小文字は',lower_text,'です。')
#print('a'>'z')
#print('123a'.isdecimal())
#string = input("文字列を入力してください:")
#if string.isdecimal():
#    print(string,"は数字です。")
#elif string.isalpha():
#    print(string,"はアルファベットです。")
#else:
#    print(string,"は数字でもアルファベットでもありません。")
#age = 11
#height = 130
#if(10 <= age) and (height >= 120):
#    print("遊園地のアトラクションに乗れます。")
#else:
#    print("遊園地のアトラクションに乗れません。")
#text = ''
#while text != 'finish':
#    text = input("文字列を入力してください（終了するには'finish'と入力）:")
#    print("入力された文字列:", text)
#print ("プログラムを終了します。")

#counter = 1
#while counter <= 5:
#    text = input("数字を入力してください:")
#    if text == '':
#        print('入力が無効です')
#        continue
#    if text == '999':
#        print("中断します")
#        break
#    number = int(text)
#    print(counter,"回目:", number*number)
#    counter = counter + 1
#
#print("終了しました")

#i = 0
#while i < 10:
#    i = i + 1
#    if i % 2 == 1:
#        continue
#    print(i, "is even")

#momo = input('ももは何個買いますか？')
#num_momo = int(momo)
#mikan = input('みかんは何個買いますか？')
#num_mikan = int(mikan)
#total_momo = num_momo * 100
#total_mikan = num_mikan * 40
#total = total_momo + total_mikan
#print('もも',num_momo,'個とみかん',num_mikan,'個の合計金額は',total,'円です。')

def fruit_price(number_of_momo, number_of_mikan):
    total_momo = number_of_momo * 100
    total_mikan = number_of_mikan * 40
    total = total_momo + total_mikan
    return total

total = fruit_price(100,200)
print('もも100個とみかん200個の合計金額は',total,'円です。')

