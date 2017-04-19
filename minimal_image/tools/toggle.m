function selected = toggle(i, selected, h_border)


if selected(i) == 0 
    selected(i)=1;
    set(h_border(i), 'color', 'g')
else
    selected(i)=0;
    set(h_border(i), 'color', 'r')
end
