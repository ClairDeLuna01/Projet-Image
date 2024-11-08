# Flags de compilation
CFLAGS = -Wall -std=c++23 -O3 -g -Wno-catch-value -Wno-switch


# Dossier d'includes
IDIR = include

# Dossier de dépendances
DEPDIR = .deps
DEPFLAGS_BASE = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

# Récupération récursive des fichiers source
rwildcard=$(foreach d,$(wildcard $(1:=/*)),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

# Fichiers source en C++ et objets correspondants
SOURCESCPP := $(call rwildcard,src,*.cpp)
OBJCPP := $(SOURCESCPP:src/%.cpp=obj/%.o)

# Fichiers source en C et objets correspondants
SOURCESC := $(call rwildcard,src,*.c)
OBJC := $(SOURCESC:src/%.c=obj/%.o)

# Fichier principal et objets associés
MAINSCPP := $(wildcard *.cpp)
OBJMAIN := $(MAINSCPP:%.cpp=obj/%.o)
MAINS := $(MAINSCPP:.cpp=)

# Liste complète des fichiers de dépendances
DEPFILES := $(OBJCPP:obj/%.o=$(DEPDIR)/%.d) $(OBJC:obj/%.o=$(DEPDIR)/%.d) $(OBJMAIN:obj/%.o=$(DEPDIR)/%.d)

# Création des dossiers obj et .deps si nécessaire
$(shell mkdir -p obj)
$(shell mkdir -p $(DEPDIR))

# Règle par défaut
default: main_rule

# Règles de compilation
obj/%.o: src/%.cpp
	g++ $(DEPFLAGS_BASE) -c $(CFLAGS) src/$*.cpp -o obj/$*.o -I$(IDIR)

obj/%.o: src/%.c
	g++ -c $(DEPFLAGS_BASE) $(CFLAGS) src/$*.c -o obj/$*.o -I$(IDIR)

obj/%.o: ./%.cpp
	g++ -c $(DEPFLAGS_BASE) $(CFLAGS) $*.cpp -o obj/$*.o -I$(IDIR)

# Lien final pour l'exécutable principal
$(MAINS): $(OBJMAIN) $(OBJCPP) $(OBJC)
	g++ $(CFLAGS) $(OBJMAIN) $(OBJCPP) $(OBJC) -o $@ -I$(IDIR)

# Inclusion des fichiers de dépendances
-include $(DEPFILES)

# Cible principale pour lancer la compilation
main_rule: $(MAINS)

# Nettoyage des fichiers générés
clean:
	rm -f obj/*.o $(MAINS)
	rm -f $(DEPDIR)/*.d
	rm -rf obj $(DEPDIR)
