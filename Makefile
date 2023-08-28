CC = g++
PROJECT = main
SRC = main.cpp
OPENMP = -fopenmp
LIBS = `pkg-config --cflags --libs opencv4`
$(PROJECT) : $(SRC)
	$(CC) $(SRC) $(OPENMP) -o $(PROJECT) $(LIBS)