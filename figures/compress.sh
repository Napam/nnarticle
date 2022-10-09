# Used for compressing gifs

compress () {
    gifsicle -v -i $1 -O3 --colors 64 --lossy=20 -o ${1%%.*}_compressed.gif
}

compress $1
