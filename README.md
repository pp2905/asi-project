# asi

## Overview

This is your new Kedro project, which was generated using `Kedro 0.18.9`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.


## Docker Compose
Run train compose with grafana
```
docker compose up
```


## Docker
Build train image
```
docker build -t train -f Dockerfile.train . 
```
Run train image
```
docker run train
```

Build make report image
```
docker build -t make-report -f Dockerfile.make-report . 
```
Run make report image
```
docker run make-report
```

Build find best model image
```
docker build -t find-best-model -f Dockerfile.find-best-model . 
```
Run find best model image
```
docker run find-best-model
```

Build presentation model prediction image
```
docker build -t presentation -f Dockerfile.presentation . 
```
Run presentation model prediction image
```
docker run find-best-model
```
