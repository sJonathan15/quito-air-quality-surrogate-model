# Datos REMMAQ

Los datos utilizados por el proyecto proceden de la Red Metropolitana de Monitoreo Atmosférico de Quito (REMMAQ), administrada por la Secretaría de Ambiente del Distrito Metropolitano de Quito. Los archivos históricos originales no se incluyen en este repositorio por su tamaño. Deben obtenerse desde los canales oficiales de datos ambientales del Municipio de Quito y emplearse respetando sus condiciones de uso.

## Archivos de entrada esperados

La aplicación de limpieza reconoce archivos CSV, XLSX o XLS. La primera columna debe contener la fecha y hora; las columnas restantes deben identificar parroquias o estaciones. Los once grupos de variables esperados son:

| Grupo | Nombres de archivo reconocidos |
|---|---|
| PM2.5 | `PM2.5.xlsx`, `PM25.xlsx`, `PM25.csv` |
| PM10 | `PM10.xlsx`, `PM10.csv` |
| O3 | `O3.xlsx`, `O3.csv` |
| CO | `CO.xlsx`, `CO.csv` |
| NO2 | `NO2.xlsx`, `NO2.csv` |
| SO2 | `SO2.xlsx`, `SO2.csv` |
| Temperatura | `TMP.xlsx`, `Temperatura.xlsx`, `Temperatura.csv` |
| Humedad | `HUM.xlsx`, `Humedad.xlsx`, `Humedad.csv` |
| Velocidad del viento | `VEL.xlsx`, `Viento_Velocidad.xlsx` |
| Dirección del viento | `DIR.xlsx`, `Viento_Direccion.xlsx` |
| Precipitación | `LLU.xlsx`, `Precipitacion.xlsx` |

Coloque los archivos originales en una misma carpeta local —por ejemplo, `datos_remmaq/`— o cárguelos desde la interfaz. Esa carpeta está excluida de Git. El sistema considera elegible una parroquia cuando dispone de al menos 10 de los 11 grupos, aunque el pipeline final depende de que PM2.5 esté presente porque la aplicación de limpieza lo usa como objetivo de preparación.

El resultado es un CSV consolidado con índice `Timestamp`, concentraciones disponibles, variables meteorológicas y características temporales. Ese CSV se carga posteriormente en la aplicación predictiva; cada entrenamiento selecciona uno de los contaminantes presentes como variable objetivo.

> El repositorio no contiene datos sintéticos ni una muestra creada artificialmente. La disponibilidad, formato y condiciones de acceso deben verificarse en el portal oficial de la Secretaría de Ambiente de Quito: <https://ambiente.quito.gob.ec/>.
