## To build the project
```
docker-compose up --build

```

## To force create new build 
```
docker-compose down --volumes --remove-orphans
docker-compose build --no-cache
docker-compose up
```