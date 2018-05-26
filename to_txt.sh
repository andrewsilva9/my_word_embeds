for d in **/*; do
	if [ ! -d "$d" ]; then
  		mv "$d" "$d.txt"
fi
done