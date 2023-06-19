expense = 8688
years = 18
lifeYears = 50
rate = 0.05
total = 0
for year in range(1, lifeYears + 1):
    if year <= years:
        total = (total + expense) * (1 + rate)
    else:
        total = total * (1+rate)
        print(year,total)
# 轻症，不如意外险
# 身故没意义，那个钱留着投资都划算
