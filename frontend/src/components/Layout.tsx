import { ThemeProvider, CssBaseline, Box, AppBar, Toolbar, Typography, Container } from '@mui/material'
import { createTheme } from '@mui/material/styles'

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#646cff',
        },
        secondary: {
            main: '#535bf2',
        },
        background: {
            default: '#242424',
            paper: '#1a1a1a',
        },
        text: {
            primary: 'rgba(255, 255, 255, 0.87)',
        },
    },
    components: {
        MuiAppBar: {
            styleOverrides: {
                root: {
                    backgroundColor: '#1a1a1a',
                },
            },
        },
    },
})

export default function Layout({ children }: { children: React.ReactNode }) {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Box sx={{ flexGrow: 1, minHeight: '100vh' }}>
                <AppBar position="static">
                    <Toolbar>
                        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                            Neural Network Playground
                        </Typography>
                    </Toolbar>
                </AppBar>
                <Container maxWidth="lg">
                    <Box sx={{ my: 4 }}>
                        <Typography
                            variant="h4"
                            component="h1"
                            gutterBottom
                            sx={{
                                fontWeight: 'bold',
                                textAlign: 'center',
                                mb: 3
                            }}
                        >
                            Training Dashboard
                        </Typography>

                        <Typography
                            variant="subtitle1"
                            color="text.secondary"
                            sx={{
                                textAlign: 'center',
                                mb: 4
                            }}
                        >
                            {`Do you know what's in your neural network? Train locally.`}
                        </Typography>
                        {children}
                    </Box>
                </Container>
            </Box>
        </ThemeProvider>
    )
}