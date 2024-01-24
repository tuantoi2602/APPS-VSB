CFLAGS+=$(shell pkg-config --cflags opencv)
LDFLAGS+=$(shell pkg-config --libs opencv)

PROGRAM=main-lcd-opencv

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM).cpp lcd_lib.cpp lcd_lib.h graph_class.hpp graph_struct.hpp
	g++ -std=c++11 -Wall $(PROGRAM).cpp lcd_lib.cpp -o $@ $(CFLAGS) $(LDFLAGS)

clean: 
	rm -rf $(PROGRAM)

