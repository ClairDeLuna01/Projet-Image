IDIR=include


default: main_rule

rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

SOURCESCPP := $(call rwildcard,src,*.cpp)
OBJCPP := $(SOURCESCPP:src/%.cpp=obj/%.o)

SOURCESC := $(call rwildcard,src,*.c)
OBJC := $(SOURCESC:src/%.c=obj/%.o)

MAINSCPP := $(wildcard *.cpp)
OBJMAIN := $(MAINSCPP:%.cpp=obj/%.o)
MAINS := $(MAINSCPP:.cpp=)



obj/%.o: src/%.cpp
	g++ -c -g src/$*.cpp -o obj/$*.o -I$(IDIR)

obj/%.o: src/%.c
	g++ -c -g src/$*.c -o obj/$*.o -I$(IDIR)

obj/%.o: ./%.cpp
	g++ -c -g $*.cpp -o obj/$*.o -I$(IDIR)

$(MAINS): $(OBJMAIN) $(OBJCPP) $(OBJC)
	g++ -g -o $@ obj/$(basename $@).o $(OBJCPP) $(OBJC) -I$(IDIR)

main_rule: $(MAINS)

clean:
	rm -f obj/*.o $(MAINS)
	rm -f obj/huffman/*.o


