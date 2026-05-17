def fizz_buzz(n):
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return str(n)
    
print(fizz_buzz(int(input("入力してください"))))

#def fizz_buzz(n, x, y):
#    if n % x == 0 and n % y == 0:
#        return 'FizzBuzz'
#    elif n % x == 0:
#        return 'Fizz'
#    elif n % y == 0:
#        return 'Buzz'
#    else:
#        return str(n)
    
