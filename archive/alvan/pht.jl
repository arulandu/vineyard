function Direction_Filtration(ordered_points, direction; out = "barcode", one_cycle = false )
	number_of_points = length(ordered_points[:,1]) #number of points
	heights = zeros(number_of_points) #empty array to be changed to heights for filtration
	
    # 1..n (points), n..2n (edges) n+i connects i <-> i+1 (cycle boundary)
    fv = zeros(2*number_of_points)

	for i in 1:number_of_points
		heights[i]= ordered_points[i,1]*direction[1] + ordered_points[i,2]*direction[2] #calculate heights in specificed direction
	end

	for i in 1:number_of_points
		fv[i]= heights[i] # for a point the filtration step is the height
	end
    
    # edge val = max(endpoints)
	for i in 1:(number_of_points-1)
		fv[(i+number_of_points)]=maximum([heights[i], heights[i+1]]) # for an edge between two adjacent points it enters when the 2nd of the two points does
	end
	
	fv[2*number_of_points] = maximum([heights[1] , heights[number_of_points]]) #last one is a special snowflake
	
    # dim of each simplex
    dv = []
    # points
	for i in 1:number_of_points
		append!(dv,0)
	end
	# edges are 1 dimensional
	for i in (1+number_of_points):(2*number_of_points)
		append!(dv,1) 
	end
    # what is the boundary of what? edge <-> pt pair
	D = zeros((2*number_of_points, 2*number_of_points))

    # i endpoint
	for i in 1:number_of_points
		D[i,(i+number_of_points)]=1
	end
	
    # i+1 endpoint
	for i in 2:(number_of_points)
		D[i, (i+number_of_points-1)]=1
	end
	D[1, (2*number_of_points)]=1
	
    # # of pts, # of edges
	ev = [number_of_points, number_of_points]
	
    # Eirene conversion
	S  = sparse(D)
	rv = S.rowval
	cp = S.colptr
    

	C = Eirene.eirene(rv=rv,cp=cp,ev=ev,fv=fv) # call
	
	if out == "barcode"
		if one_cycle == true
			return barcode(C, dim=0), maximum(heights)
		else
			return barcode(C, dim=0)
		end
	else
		if one_cycle == true
			return C, maximum(heights)
		else
			return C
		end
	end
end #Direction_Filtration