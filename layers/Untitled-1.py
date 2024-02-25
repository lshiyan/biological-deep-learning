import math

def is_prime(n):
    if n==1:
        return False
    if n==2:
        return True
    if n==3:
        return True
    if n==4:
        return False
    for i in range(2,n//2+1):
        if n%i==0:
            return False
    return True

        
def goldbach(x):
    if x==4:
        print("4 has 1 representation(s)")
        print("2+2")
    else:
        pairs=[]
        count=0
        for i in range(2,x//2+1):
            if is_prime(i) and is_prime(x-i):
                pairs.append([i,x-i])
                count+=1
        print("%d has %d representation(s)" %(x, count))
        for ele in pairs:
            print("%d+%d" %(ele[0],ele[1]))
        
inp=input()
first=True
for ele in inp.split(" "):
    if first:
        first=False
    else:
        goldbach(int(ele))
        print("")
        
if __name__=="__main__":
    goldbach(26)
    