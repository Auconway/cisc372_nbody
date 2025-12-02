FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile
NVCC = nvcc

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
compute.o: compute.cu config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -c $< -o $@
clean:
	rm -f *.o nbody
	