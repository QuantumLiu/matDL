function fsmgospel=Christ2FSM(bibel)
dict{1}={'god','God','LORD','Lord','lord','holy','Holy',' Amen','heaven','Heaven','hell','Hell','angle','Angle'...
    ,'demon','Demon','christ','Christ','water','Water'};
dict{2}={'monster','Monster','FSM','FSM','FSM','yummy','Yummy',' RAmen','plate','Plate','sewer','Sewer','ball','Ball'...
    ,'fork','Fork','pasta','Pasta','soip','Soup'};
if ~length(dict{1})==length(dict{2})
    error('Keywords dict length unmatched');
end
fsmgospel=bibel;
for i=1:length(dict{1})
    old=dict{1}{i};
    new=dict{2}{i};
    fsmgospel=strrep(fsmgospel,old,new);
end