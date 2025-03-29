CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2
LDFLAGS = -pthread

SRCS = kmeans_clustering.cpp vector_cluster_store.cpp test_cluster_store.cpp
OBJS = $(SRCS:.cpp=.o)
HEADERS = clustering_interface.h kmeans_clustering.h vector_cluster_store.h

.PHONY: all clean

all: test_cluster_store

test_cluster_store: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o test_cluster_store
