LUA code:
```lua
local missions = ScenEdit_GetMissions( 'United States' )
print(missions)
```

Output:
```lua
{ [1] = mission {
 guid = 'cabeaa3f-6594-495a-822d-dee0c33b91da', 
 name = 'EW', 
 side = 'United States', 
 type = 'Strike', 
 subtype = 'Air Intercept', 
 isactive = 'True', 
 starttime = '', 
 endtime = '', 
 SISH = 'False', 
 aar = 'table',
 unitlist = 'table',
} }