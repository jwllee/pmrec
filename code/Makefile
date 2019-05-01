DIR=$(shell pwd)
DATADIR=$(shell dirname "$(shell pwd)")/data

build:
	docker build -t pmrec .

run: build
	docker run -it \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $(DIR):/home/pmrec/code \
		-v $(DATADIR):/home/pmrec/data \
		-e DISPLAY=unix$(DISPLAY) \
		--device /dev/snd:/dev/snd \
		pmrec /bin/bash
