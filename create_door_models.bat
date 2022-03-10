@ECHO OFF
SETLOCAL EnableDelayedExpansion
SETLOCAL
SET prog=%0
SET args=%*
IF !args! == /? (
    CALL :usage
    ECHO.
    ECHO The first form will call the following command:
    ECHO python create_door_models.py -o assets\NAMESPACE\models\block\ assets\true3d\templates\door\ NAMESPACE:NAME ARGS
    EXIT /B
)
IF ["%~1"]==[""] (
    CALL :usage
    ECHO Missing argument #1: NAMESPACE
    EXIT /B 1
)
IF ["%~2"]==[""] (
    CALL :usage
    ECHO Missing argument #2: NAME
    EXIT /B 1
)
SET "namespace=%~1"
SET "name=%~2"
SHIFT
SHIFT
@ECHO ON
python create_door_models.py -o assets\%namespace%\models\block\ assets\true3d\templates\door\ %namespace%:%name% %1 %2 %3 %4 %5 %6 %7 %8 %9
@ECHO OFF
EXIT /B
:usage
ECHO usage: %prog% NAMESPACE NAME [ARGS...]
ECHO OR %prog% /?
EXIT /B
