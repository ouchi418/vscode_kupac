def fizz_buzz(n, x, y):
    if n % x == 0 and n % y == 0:
        return 'FizzBuzz'
    elif n % x == 0:
        return 'Fizz'
    elif n % y == 0:
        return 'Buzz'
    else:
        return str(n)
