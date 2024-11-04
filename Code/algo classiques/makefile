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

DEPDIR := .deps
DEPFLAGS_BASE = -MT $@ -MMD -MP -MF $(DEPDIR)
DEPFLAGS = $(DEPFLAGS_BASE)/$*.d
DEPFLAGSMAIN = $(DEPFLAGS_BASE)/main.d


obj/%.o: src/%.cpp
	g++ $(DEPFLAGS_BASE) $(DEPFLAGS) -c -g src/$*.cpp -o obj/$*.o -I$(IDIR)

obj/%.o: src/%.c
	g++ -c $(DEPFLAGS_BASE) $(DEPFLAGS) -g src/$*.c -o obj/$*.o -I$(IDIR)

obj/%.o: ./%.cpp
	g++ -c $(DEPFLAGS_BASE) $(DEPFLAGS) -g $*.cpp -o obj/$*.o -I$(IDIR)

$(MAINS): $(OBJMAIN) $(OBJCPP) $(OBJC)
	g++ -g $(DEPFLAGS_BASE) $(DEPFLAGSMAIN) -o $@ obj/$(basename $@).o $(OBJCPP) $(OBJC) -I$(IDIR)

$(DEPDIR): ; @mkdir $@

DEPFILES := $(SOURCES:$(SDIR)/%.cpp=$(DEPDIR)/%.d)
DEPFILES += $(DEPDIR)/main.d
# $(info $(DEPFILES))
$(DEPFILES):


include $(wildcard $(DEPFILES))

main_rule: $(MAINS)

clean:
	rm -f obj/*.o $(MAINS)
	rm -f obj/*.d $(DEPDIR)/*.d

	


