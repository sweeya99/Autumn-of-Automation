
def maxProfit(price, start, end): 

	# If the stocks can't be bought 
	if (end <= start): 
		return 0; 

	# Initialise the profit 
	profit = 0; 

	# The day at which the stock 
	# must be bought 
	for i in range(start, end, 1): 

		# The day at which the 
		# stock must be sold 
		for j in range(i+1, end+1): 

			# If byuing the stock at ith day and 
			# selling it at jth day is profitable 
			if (price[j] > price[i]): 
				
				# Update the current profit 
				curr_profit = (price[j] - price[i] +
							maxProfit(price, start, i - 1)+ 
							maxProfit(price, j + 1, end) )

				# Update the maximum profit so far 
				profit = max(profit, curr_profit); 

	return profit



if __name__ == '__main__': 

	price = [1100, 5000, 100, 4000, 10] 
	n = len(price)

	print(maxProfit(price, 0, n - 1))


