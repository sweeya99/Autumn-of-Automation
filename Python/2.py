num = int (input("enter a num : "))
def func(num):
        num+=1
        num_arr = list((int(x) for x in str(num)))
        if num_arr==num_arr[::-1]:
            return int(''.join(str(x) for x in num_arr))
        arr_len = len(num_arr)
        if arr_len%2==0:
            mid = arr_len//2-1
        else:
            mid = arr_len//2
        temp=0
        for i in range(mid,-1,-1):
            if num_arr[i]!=num_arr[arr_len-i-1]:
                temp = i
                break
        if int(''.join(str(x) for x in num_arr[:temp+1][::-1])) > int(''.join(str(x) for x in num_arr[arr_len-temp-1:])):
            for i in range(mid+1):
                num_arr[arr_len-i-1] = num_arr[i]
        else:
            num_arr[mid]+=1
            for i in range(mid,-1,-1):
                if num_arr[i]==10:
                    num_arr[i]=0
                    num_arr[i-1]+=1
            for i in range(mid+1):
                num_arr[arr_len-i-1] = num_arr[i]
        return int(''.join(str(x) for x in num_arr))    

x = func(num)
print(x)
